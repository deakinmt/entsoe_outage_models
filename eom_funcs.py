import os, sys, csv, pickle, requests, types, zipfile
import numpy as np
from datetime import timedelta,datetime
from functools import lru_cache
from pprint import pprint
requests.Session()
from timeit import default_timer as timer
from collections import OrderedDict as odict
from bunch import Bunch
import xml.etree.ElementTree as ET
from importlib import reload
from copy import deepcopy
from scipy import sparse
import quantecon as qe
from dateutil.tz import gettz
from pytz import country_timezones

from eom_utils import repl_list, fn_root, data2csv, structDict, sctXt,\
        tDict2stamps, boxplotqntls, nanlt, nangt, plotEdf, cdf2qntls, rerr
exec(repl_list)
from entsoe_py_data import BIDDING_ZONES, PSRTYPE_MAPPINGS, \
                            PSRTYPE_MAPPINGS_INTER
from progress.bar import Bar

from numpy.random import MT19937, Generator, SeedSequence
def rngSeed(seed=0,):
    """Seed a random number generator at seed.
    
    Used seed = None for a random state.
    """
    if seed is None:
        return Generator(MT19937(SeedSequence(seed)))
    else:
        return Generator(MT19937())

rng = rngSeed()

# For the list of attributes, see 'ENTSO-E TRANSPARENCY PLATFORM
# DATA EXTRACTION PROCESS IMPLEMENTATION GUIDE'.

# # For debugging XML structures:
# for child in ts:
    # print(child.tag,child.attrib,child.tail,child.text)

# These domain codes for each country/zone are found within
# 'Transparency Platform restful API - user guide'. (Also
# this has been copied into the appendixA.xlsx spreadsheet.)
if '__file__' in dir():
    with open(os.path.join(os.path.dirname(__file__),
                                    'apiDoc','bzAreas.csv'),'r') as file:
        bzs = list(csv.reader(file,delimiter='\t'))
else:
    with open(os.path.join(sys.path[0],'apiDoc','bzAreas.csv'),'r') as file:
        bzs = list(csv.reader(file,delimiter='\t'))


from download_config import token
url0 = r'https://transparency.entsoe.eu/api?securityToken='+token

# B0630_BTA60, B0630_BTA61 are to be read as '*_BT*'; e.g., the business 
# type of 'B0630_BTA61' being A61 (see processXml).
reqInfo = {
            'dap':{
                   'id':0,
                   'reqOpts':[None],
                   'dataName':'price.amount',
                   'unitHead':'currency_Unit.name'
                   },
            'actlTotLd':{
                   'id':1,
                   'reqOpts':[None],
                   'dataName':'quantity',
                   'unitHead':'quantity_Measure_Unit.name'
                   },
            'fcstTotLd':{
                   'id':2,
                   'reqOpts':[None],
                   'dataName':'quantity',
                   'unitHead':'quantity_Measure_Unit.name'
                   },
            'wNsF':{
                   'id':3,
                   'reqOpts':['PsrType=B16','PsrType=B18','PsrType=B19'],
                   'dataName':'quantity',
                   'unitHead':'quantity_Measure_Unit.name'
                   },
            'B0630_BTA60':{
                   'id':4,
                   'reqOpts':[None],
                   'dataName':'quantity',
                   'unitHead':'quantity_Measure_Unit.name'
                   },
            'B0630_BTA61':{
                   'id':5,
                   'reqOpts':[None],
                   'dataName':'quantity',
                   'unitHead':'quantity_Measure_Unit.name'
                   },
            'wNs':{
                   'id':6,
                   'reqOpts':['PsrType=B16','PsrType=B18','PsrType=B19'],
                   'dataName':'quantity',
                   'unitHead':'quantity_Measure_Unit.name'
                   },
            'unvblGp':{
                    'id':7,
                    'reqOpts':[None],
                    'dataName':'',
                    'unitHead':''
                    },
            'icppu':{
                    'id':8,
                    'reqOpts':[None],
                    'dataName':'',
                    'unitHead':''
                    },
            'png':{
                    'id':9,
                    'reqOpts':[None],
                    'dataName':'',
                    'unitHead':''
                    },
            }

#key/value pairs for availability data.
APs0 = 'production_RegisteredResource.pSRType.powerSystemResources.mRID'
mridData = {
'nomP':'production_RegisteredResource.pSRType.powerSystemResources.nominalP',
'mu':'quantity_Measure_Unit.name',
'psrName':'production_RegisteredResource.pSRType.powerSystemResources.name',
'psrType':'production_RegisteredResource.pSRType.psrType',
'locName':'production_RegisteredResource.location.name',
'prrName':'production_RegisteredResource.name',
'prrMrid':'production_RegisteredResource.mRID',
}

APschema = """Availability Periods data schema:
[1]changes
-[1]createdDateTime
-[1]docStatus
-[1]businessType
-[1]revisionNumber
- TimeSeries/
--[0]production_RegisteredResource.pSRType.powerSystemResources.mRID
--[m]production_RegisteredResource.pSRType.powerSystemResources.nominalP
--[m]quantity_Measure_Unit.name
--[m]production_RegisteredResource.pSRType.powerSystemResources.name
--[m]production_RegisteredResource.pSRType.psrType
--[m]production_RegisteredResource.location.name
--[m]production_RegisteredResource.name
--[m]production_RegisteredResource.mRID
-- Available_Period/
--- timeInterval/
----[2]start
----[2]end
--- Point/
----[2]quantity
"""

genXmlData = {
        'mrid':['registeredResource.mRID',],
        'name':['registeredResource.name'],
        'gType':['MktPSRType','psrType'], # generation type
        'cap':['Period','Point','quantity',],
        'unit':['quantity_Measure_Unit.name',],
        'tStt':['Period','timeInterval','start',],
        'tEnd':['Period','timeInterval','end',],
            }


def getCodes(cds=None):
    """Get the key:value bidding codes for cds.
    
    Inputs
    ---
    cds - the country codes. If None, uses a nominal set; ir 'CH'
    
    """
    if cds is None:
        # I0 required as SEM and IE only do different stuff for some data types
        cds = ['NO-2','DK-1','NL','FR','BE','IE','GB','I0']
    elif cds=='CH':
        cds = ['NO-1','NO-2','NO-3','NO-4','NO-5',
                'DK-1','DK-2','NL','FR','BE','IE','GB','I0','DE','DE18','ES',]
    
    codes = {key:BIDDING_ZONES[key] for key in cds}
    return codes

def getUrl(reqType,code,yr,opts=None):
    """Create the URL to request from the ENTSO-e API.
    
    Inputs
    ---
    reqType - request type
    code - BZ code for the country
    yr - the year to use
    opts - options to pass into getUrlBase
    
    """
    t0 = 'periodStart='+str(yr)+'01010000'
    t1 = 'periodEnd='+str(yr+1)+'01010000'
    urlBase = getUrlBase(reqType,code,opts)
    url = urlBase if reqType=='png' else '&'.join([urlBase,t0,t1])
    return url
    
def getUrlBase(reqType,code,opts):
    """Build a url based on req type, country code and opt params.
    
    opts should be input as a list of strings, to be 
    appended to the url one by one. For example, it could be
    the psr type for 'wNsF'. Only used by some options though:
    - wNsF, wNs
    
    """
    if reqType=='dap':
        # day ahead price
        urlBase = '&'.join([url0,
                            'documentType=A44',
                            'In_Domain='+code,
                            'Out_Domain='+code,
                            ])
    elif reqType=='actlTotLd':
        urlBase = '&'.join([url0,
                            'documentType=A65',
                            'ProcessType=A16',
                            'outBiddingZone_Domain='+code
                            ])
    elif reqType=='fcstTotLd':
        urlBase = '&'.join([url0,
                            'documentType=A65',
                            'ProcessType=A01',
                            'outBiddingZone_Domain='+code
                            ])
    elif reqType=='wNsF':
        # wind 'n' solar FORECAST
        urlBase = '&'.join([url0,
                            'documentType=A69',
                            'ProcessType=A01',
                            'In_Domain='+code,
                            ])
        if opts is not None:
            urlBase = '&'.join([urlBase]+opts)
    elif reqType in ['B0630_BTA60','B0630_BTA61']:
        urlBase = '&'.join([url0,
                            'documentType=A65',
                            'ProcessType=A31',
                            'outBiddingZone_Domain='+code
                            ])
    elif reqType=='wNs':
        # wind 'n' solar OUTTURN
        urlBase = '&'.join([url0,
                            'documentType=A74',
                            'ProcessType=A16',
                            'In_Domain='+code,
                            ])
        if opts is not None:
            urlBase = '&'.join([urlBase]+opts)
    elif reqType in ['unvblGp',]:
        # generation/production unavailabilities
        urlBase = '&'.join([url0,
                            'documentType=A80', # generation unavailability
                            'ProcessType=A53', # planned maintenance
                            'BiddingZone_Domain='+code,
                            ])
        if opts is not None:
            urlBase = '&'.join([urlBase]+opts)
    elif reqType in ['icppu',]:
        urlBase = '&'.join([url0,
                            'documentType=A71', # generation unavailability
                            'ProcessType=A33',
                            'In_Domain='+code,
                            ])
    elif reqType in ['png',]: # see 4.5. Master Data in web api guide.
        urlBase = '&'.join([url0,
                            'documentType=A95',
                            'businessType=B11',
                            'BiddingZone_Domain='+code,
                            'implementation_DateAndOrTime=__datehere__',
                            ])
    return urlBase

@lru_cache(maxsize=64)
def reqData(url):
    """lru_cache (size 64) to avoid multiple unneccessary calls.
    
    If a 400 error is returned, this prints the error.
    """
    TIMEOUT=60
    print('Request data...')
    r = requests.get(url,timeout=TIMEOUT)
    if r.status_code==400:
        print('HTTP error for url:')
        pprint(url.split('&'))
        print('\n'+'-'*12+'\n'+r.content.decode('utf-8')+'\n'+'-'*12)
    print('\t\t...returned:',r)
    return r

def processXml(root,reqType):
    """A basic method to process the XML data from entso-e.
    
    Note that, for now, it is MOSTLY suitable to be used
    with xmls that have a single data point per timestamp
    (so, for example, wNsF should choose one psr type).
    
    Use a name with *_BTA60 to choose a business type.
    
    Inputs
    ---
    root - an ETnse XML root (NOT a simple root)
    reqType - the required data type string
    
    Returns
    ---
    (tsOut, curve) - a clock (with uniform timesteps) and curve (with nans)
    unit - the units
    """
    timeSeries = root.findall('TimeSeries')
    
    # First, return if no data (print to console if this happens)
    if len(timeSeries)==0:
        print('Empty time series, returning empty values.')
        return (np.empty((0),dtype=np.datetime64),np.empty((0))), None
    
    # If looking at a specific business type, filter on those time series
    if len(reqType.split('_'))>1:
        reqT0,bt = reqType.split('_BT')
        timeSeries = [ts for ts in timeSeries \
                            if ts.find('businessType').text==bt]
    
    # Util function to determine the time 
    trt2dt = lambda trt: timedelta(1,) if trt[1]=='D' else \
            timedelta(0,60*int(''.join(filter(str.isdigit,trt,))))
    
    crvs, tss, units, timeRes = mtList(4)
    for ts in timeSeries:
        prd = ts.find('Period')
        
        # Pull out the fixed data for the whole period
        units.append(ts.find(reqInfo[reqType]['unitHead']).text)
        tr = trt2dt( prd.find('resolution').text )
        ti_str = prd.find('timeInterval').find('start').text
        
        # Check times are in UTC and get time
        assert(ti_str[-1]=='Z')
        ti = datetime.fromisoformat(ti_str[:-1])
        
        # Build the curves
        for pt in prd.findall('Point'):
            timeRes.append(tr)
            ii = int(pt.find('position').text) - 1
            crvs.append(float(pt.find(reqInfo[reqType]['dataName']).text))
            tss.append(ti+ii*timeRes[-1])
    
    # Check there is only one type of unit
    assert(len(set(units))==1)
    
    # Check that we only have up to two time resolutions
    assert(len(set(timeRes))<=2)
    if len(set(timeRes))==2:
        assert((max(timeRes)/min(timeRes)).is_integer)
        print(f'Min/max resolutions: {min(timeRes)}, {max(timeRes)}')
    
    # Build a clock
    dt = min(timeRes)
    tsOut = np.arange(min(tss),max(tss)+timeRes[tss.index(max(tss))],dt,
                                                            dtype=datetime)
    
    # Build the time series curve to match
    curve = np.nan*np.zeros(len(tsOut))
    for i,(t,tr,) in enumerate(zip(tss,timeRes)):
        i0 = (t - tsOut[0])//dt
        curve[slice(i0,i0+(tr//dt))]=crvs[i]
    
    return (tsOut,curve),set(units).pop()

class ETnse(ET.Element,):
    """Overloaded version of Element Tree ET to avoid having to use xmlns.
    
    """
    def __init__(self,root,):
        self._xmlns = (root.tag).split('}')[0]+'}' # xmlns
        self._root = root
        self.set_data_descriptors()
    
    def set_data_descriptors(self,):
        """Set text, tag, etc from self._root"""
        for k,v in type(self._root).__dict__.items():
            if type(v) is types.GetSetDescriptorType:
                self.__setattr__(k,self._root.__getattribute__(k),)
    
    def __repr__(self,):
        return self._root.__repr__()
    
    def getchildren(self,):
        """ETnse overload of getchildren(root)."""
        return [r.tag.replace(self._xmlns,'') 
                            for r in self._root.getchildren()]
    
    def findall(self,ss):
        """ETnse overload of findall(ss) with _xmlns."""
        return [ETnse(r) for r in self._root.findall(self._xmlns+ss)]
    
    def find(self,ss):
        """ETnse overload of find(ss) with _xmlns."""
        rootfind = self._root.find(self._xmlns+ss)
        if rootfind is None:
            return None
        else:
            return ETnse(rootfind)

def check_ssend_flg(stts,ends):
    """Check the starts are aligned, then if the starts/ends overlap."""
    if not all([s0<=s1 for s0,s1 in zip(stts[:-1],stts[1:])]):
        return False
    else:
        return all([e<=s for s,e in zip(stts[1:],ends[:-1])])

def contiguous_duplicates(stts,ends,vals,nomps,dlo,dhi,):
    """Create a contiguous equivalent outage when there is a stepped curve.
    
    Stts, ends must be such that we can just take the difference of these
    and multiply by 'val' to get the output.
    
    Inputs
    ---
    stts, ends - start & end times
    vals - the power values between start and end times
    nomps - the nominal power of the device
    
    Returns
    ---
    vvl - the equivalent loading level, in units of vals/nomps (MW likely)
    """
    if not check_ssend_flg(stts,ends):
        print('A clash in the non-unique APs in contiguous duplicates!')
        return np.nan, np.nan
    
    dt = dhi - dlo
    
    ends_ = npa([min(e,dhi) for e in ends])
    stts_ = npa([max(e,dlo) for e in stts])
    
    return ((ends_-stts_)/dt).dot(vals)

def minmax_reconciliation(stts,ends,vals,nomps,dlo,dhi,):
    """Find the maximum and minimum possible availabilities of generators.
    
    Used when other methods have failed.
    
    NB: assumes one minute time resolution.
    """
    dt = timedelta(0,60,)
    nt = (dhi-dlo)//dt
    
    vmat = np.nan*np.ones((len(stts),nt,))
    i0s = [max(0,            (s-dlo)//dt) for s in stts]
    i1s = [min((dhi-dlo)//dt,(s-dlo)//dt) for s in ends]
    
    # Assign the values to vmat
    _ = [vmat[ii].__setitem__(slice(i0,i1),v) 
                        for ii,(i0,i1,v) in enumerate(zip(i0s,i1s,vals))]
    
    # Create the max and min versions of the output
    vmax, vmin = [np.nanmin(np.c_[np.ones(nt)*nomps, mnmx(vmat,axis=0)],axis=1)
                                        for mnmx in [np.nanmax,np.nanmin]]
    
    return np.mean(vmax), np.mean(vmin)

def load_aps(cc,sd,rerun=True,save=False,):
    """Load the availability periods list APs for country cc.
    
    Method:
    - Initialise a dict for storing the information (APs)
    - Read each entry of the zipped XML data
    - Append each generator with an outage at the relevant datetime kk
    
    Note that the keys for the dict kk are in UTC, where the dates that are
    saved in the outage files are in local time; so, the datetime of the keys
    do NOT match up with the datetimes of the saved zip files.
    
    Inputs
    ---
    cc - country to load
    sd - save directory
    rerun - if False, then simply load the data from 
    
    Returns
    ---
    APs - the availability period dict-of-dicts for reports
    mm - dict of all mmRids of the generators
    kks - string format of kksD as a key for APs
    kksD - the sorted start datetimes of all of the APs keys
    
    """
    ld_ = os.path.join(sd,'outage_data')
    sd_ = os.path.join(sd,'output_cache','APs',)
    _ = os.mkdir(sd_) if not os.path.exists(sd_) else None
    fn_ = os.path.join(sd_,f'{cc}_APs.pkl')
    
    # If not rerun, reload the cached data and return
    if not rerun:
        with open(fn_,'rb') as file:
            data = pickle.load(file)
        print(data['readme'])
        return data['APs'], data['mm'], data['kks'], data['kksD']
    
    t0 = timer()
    ap2clk = lambda ap,ss: datetime.fromisoformat(
            ap.find('timeInterval').find(ss).text.replace('Z',':00'))
    
    # Approach: a day has a list of mRIDs. If a mRID does not have it's key
    # information in mRIDdict, then this is added. Then, for each availability
    # period, the data is saved.
    mm = {} # mRIDmaster dict
    
    print(f'Loading {cc} APs...')
    fns = get_path_files(os.path.join(ld_,cc))
    APs = {} # each file
    with Bar('Load APs', suffix='%(percent).1f%% - %(eta)ds',
                                                    max=len(fns)) as bar:
        for fn in fns:
            # First check the zip file is a zip file and not an error .txt
            if not zipfile.is_zipfile(fn):
                # Double check this is an xml text file, as expected
                with open(fn,'r') as ff:
                    ET.XML(ff.read())
                
                bar.next()
                continue
            
            # This finds the dict key as the date
            kk_ = fn.split('_')[-3]
            kk = get_utc_kk(kk_,cc,)
            
            # If not existing yet, add new one.
            if kk not in APs.keys():
                APs[kk] = {}
            
            with zipfile.ZipFile(fn, 'r') as zip_ref:
                for zr in zip_ref.infolist():
                    # Get an XML representation of the entry
                    root = ET.XML(zip_ref.read(zr))
                    rr = ETnse(root)
                    ts = rr.find('TimeSeries')
                    aps = ts.findall('Available_Period')
                    
                    # Update mm
                    id = ts.find(APs0).text
                    if id not in mm.keys():
                        mm[id] = {}
                        for k,v in mridData.items():
                            try:
                                mm[id][k] = ts.find(v).text
                            except:
                                mm[id][k] = None
                    
                    if not (id in APs[kk].keys()):
                        APs[kk][id] = []
                    
                    ds = None if rr.find('docStatus') is None \
                            else rr.find('docStatus').find('value').text
                    
                    APs[kk][id].append(
                        {
                        'createdDateTime':datetime.fromisoformat(
                                        rr.find('createdDateTime').text[:-1]),
                        'docStatus':ds,
                        'businessType':ts.find('businessType').text,
                        'revisionNumber':int(rr.find('revisionNumber').text),
                        'changes':'CHANGES' in zr.filename,
                        'data':[{
                            'start':ap2clk(ap,'start'),
                            'end':ap2clk(ap,'end'),
                            'val':float(ap.find('Point').find('quantity').text),
                                } for ap in aps]
                        }
                    )
            bar.next()
    
    print(f'APs for {cc} loaded.')
    
    # Build a linearly increasing clock and dates
    kksD = np.sort(np.array([s2d(kk) for kk in APs.keys()]))
    kks = [d2s(kk) for kk in kksD]
    
    # Finally, if we are saving the data, put in a dict and save
    if save:
        nl = '\n'
        readme = f'{cc} APs created at {datetime.today().isoformat()}{nl}'\
                + f'- Start date: {min(APs.keys())}{nl}'\
                + f'- End date: {max(APs.keys())}{nl}'\
                + f'- Time taken to build: {timer() - t0:1g}s{nl}{nl}'\
                + f'See help(eomf.load_aps) for description of contents.{nl}'
        
        with open(fn_,'wb') as file:
            pickle.dump({'APs':APs,'mm':mm,'kks':kks,
                                       'kksD':kksD,'readme':readme},file)
    
    return APs, mm, kks, kksD

def load_dps(dstart,dend,cc,sd,rerun=True,save=False,ei=True,):
    """Block process the APs to find the dps values for each country.
    
    Inputs: see help(eomf.load_aps) for inputs / help(eomf.block_process_aps)
    
    The bulk of the work is done by "block_process_aps" which, in-turn, 
    calls "process_aps" for each of the hourly time periods.
    
    This also saves moX, moXx. Load these with using load_mouts.
    
    Returns
    ---
    drange - the clock
    dpsX - the forced / planned / total outages
    dpsXx - the forced / planned / total outages max additional outages
    """
    sd_ = os.path.join(sd,'output_cache','DPs',)
    _ = os.mkdir(sd_) if not os.path.exists(sd_) else None
    fn_ = os.path.join(sd_,f'{cc}_APs{d2s(dstart)}-{d2s(dend)}.pkl')
    
    # If not rerun, reload the cached data and return
    if not rerun:
        with open(fn_,'rb') as file:
            data = pickle.load(file)
        print(data['readme'])
        return [data[k] for k in ['drange','dpsX','dpsXx',]]
    
    print(f'========== Processing {cc} APs...')
    APs, mm, kks, kksD = load_aps(cc,sd,rerun=False,save=False)
    
    # Load in the PNG dict for more up-to-date generator data
    fns = [fn for fn in get_path_files(os.path.join(sd,'png',cc,))
                                    if '2020-12-01.csv' in fn]
    heads, datas = listT([csvIn(fn) for fn in fns])
    assert(all([heads[0]==h for h in heads]))
    idxM,idxP = [heads[0].index(h) for h in ['mRID','nomP']]
    pngNomP = {d[idxM]:float(d[idxP]) for data in datas for d in data}
    
    # Checksum if wanted; then, update mm nomP to the new values
    neqs = [(mm[k]['nomP'],pngNomP.get(k,-1)) for k in mm.keys() \
                if mm[k]['nomP']!=str(int(pngNomP.get(k,-1)))]
    _ = [mm[k].__setitem__('nomP',pngNomP.get(k,mm[k]['nomP'])) 
                                                        for k in mm.keys()]
    
    # Run!
    t0 = timer()
    drange,dpsX,dpsXx,moX,moXx \
                = block_process_aps(dstart,dend,kksD,APs,mm,kks,ei=ei,)
    
    if save:
        nl = '\n'
        readme = f'{cc} DPS created at {datetime.today().isoformat()}{nl}'\
                + f'- Exclude intermittent generation: {ei}{nl}'\
                + f'- Start date: {dstart}{nl}'\
                + f'- End date: {dend}{nl}'\
                + f'- Time taken to build: {timer() - t0:1g}s{nl}'
        
        with open(fn_,'wb') as file:
            pickle.dump({'drange':drange,'dpsX':dpsX,'dpsXx':dpsXx,
                                                'readme':readme},file)
        
        with open(fn_.replace('.pkl','_ivdl.pkl'),'wb') as file:
            pickle.dump({'drange':drange,'moX':moX,'moXx':moXx,
                                                'readme':readme},file)
    
    return drange, dpsX, dpsXx

def load_mouts(dstart,dend,cc,sd,):
    """Load the individual generator outputs moF, moP.
    
    The data for this is saved in eomf.load_dps (with 'rerun' & 'save' on).
    
    NB: the order returned is Planned THEN Forced.
    
    Returns
    ---
    drange - the datetimes in moF, moP
    moX - the planned/forced/total outages
    moXx - the planned/forced/total possible additional outages
    mo2x - util func for converting moX, moXx to time series
    """
    sd_ = os.path.join(sd,'output_cache','DPs',)
    fn_ = os.path.join(sd_,f'{cc}_APs{d2s(dstart)}-{d2s(dend)}_ivdl.pkl')
    with open(fn_,'rb') as file:
        data = pickle.load(file)
    
    print(data['readme'])
    dout = [data[k] for k in ['drange','moX','moXx',]]
    
    def mo2x(k):
        """Convert the sparse moP, moF representation to a full time series.
        
        Defined in the eomf.load_mouts function to use the correct dict/drange.
        
        Input: mmrid - generator key for moF/moP.
        Returns: xx - (nt,2)-sized np array, with ith row as [moF,moP,moFx,moPx]
        """
        xx = np.zeros((len(dout[0]),6))
        xvals = [dout[ii][j] for ii in [1,2] for j in ['p','f','t',]]
        _ = [[xx[:,ii].__setitem__(v[0],v[1]) for v in mo[k]]
                                            for ii,mo in enumerate(xvals)]
        return xx
    
    return dout + [mo2x]

def process_aps(APsK,dlo,dhi,mm,excl_intermittent=True,):
    """Use APs to find the forced and unforced outages.
    
    This is the core of the method for determining country-level outages.
    
    Approach.
    ---
    -> First, remove any reports which do not have good doc status ('A09' - 
        see [1], 'A.8. DocStatus'), 
    -> All other reports are put into a single list of triplets, containing
        info on (mrid, APs report no, data no).
    -> All generators which only have a single report in the given time period
        are first processed, with the outage value calculated by taking the
        difference between the mrid's nominal power and the power at that time.
        -> Note: in these cases, the outage value is taken as an average (see
            get_vm0) - e.g., if the outage is at 0% for 15 mins of an hour, but
            is otherwise available, then this is counted as 75%.
    -> Generators which do not have just one outage are then processed:
        -> Where the time periods do not overlap, the outage is simply a 
            weighted average, with no ambiguity.
        -> For those outages which are ambiguous, the maximum and minimum 
            possible outages are calculated; the minimum outage is saved with 
            the maximum additional outage going into vvx.
    
    Inputs
    ---
    APsK - availabilities dict value that will be used from dlo to dhi. Is
            the 'kth' item of an APs dict [which, may have only planned or 
            forced outages, dependent on usage].
    dlo - the start time
    dhi - the end time
    mm - the dict of generator data
    excl_intermittent - bool (to exclude intermittent in PSRTYPE_MAPPINGS_INTER)
    
    Outputs [I think these are not independent...?]
    ---
    vvmult - a list of multipliers (1 if outage during whole period)
    vvals - the value reported in the entsoe files
    nomps - the corresponding nominal power
    vvx - additional outages when there are clashes.
    
    [1] https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html . Accessed 18/10/21
    
    """
    # dhi = dlo+timedelta(0,1800) if dhi is None else dhi
    mmRdr = list(APsK.keys())
    
    # Util funcs - m,j,k: mRID, aps item no., report data idx
    aps_mj = lambda m,j:     APsK[m][j]
    get_data = lambda m,j,k: APsK[m][j]['data'][k]
    get_vm0 = lambda m,j,k: ( min(get_data(m,j,k)['end'],dhi) -
                            max(get_data(m,j,k)['start'],dlo))/(dhi - dlo)
    nom2float = lambda nomP: np.nan if nomP is None else float(nomP)

    # First, pull out all of data into a list (if the docStatus + dates are ok)
    ap_out = []
    for m in mmRdr:
        # Only use generators that are not wind/solar/hydro ror
        if mm[m]['psrType'] in PSRTYPE_MAPPINGS_INTER.keys()\
                    and excl_intermittent:
            continue
        
        for j,aps in enumerate(APsK[m]):
            if aps['docStatus']!='A09':
                for k,dd in enumerate(aps['data']):
                    if (dhi>dd['start'] and dlo<dd['end']):
                        if get_data(m,j,k)['val']>1.333*float(mm[m]['nomP']):
                            print(f'\nIgnoring {m} at {dlo} - output too high.')
                            continue
                        
                        ap_out.append([m,j,k,])
    
    # Find simple (unique) values records, and then duplicates
    mmsel = [a[0] for a in ap_out]
    unq_idxs = [i for i,m_ in enumerate(mmsel) if mmsel.count(m_)==1]
    mm_nunq = list(set([m_ for m_ in mmsel if mmsel.count(m_)!=1]))
    mout = [mmsel[i] for i in unq_idxs] + mm_nunq
    
    # Build a dict of indexes for use with the non-unique generators
    mm_idx_dict = mtDict(mmsel)
    _ = [mm_idx_dict[a_[0]].append(i) for i,a_ in enumerate(ap_out)]
    
    # Get the time multiplier, nom power, and AP value
    vvmult = [get_vm0(*ap_out[i]) for i in unq_idxs]
    vvals = [get_data(*ap_out[i])['val'] for i in unq_idxs]
    nomps = [nom2float(mm[mmsel[i]]['nomP']) for i in unq_idxs]
    vvx = [0]*len(unq_idxs)
    
    # Deal with any duplicates
    for nunq in mm_nunq:
        # First, pull out the indexes in ap_out and nominal power
        idxs = mm_idx_dict[nunq]
        nomps.append(nom2float(mm[mmsel[idxs[0]]]['nomP']))
        
        # Then, create lists of the data used to create the value:
        vals, stts, ends = [[get_data(*ap_out[i])[kk] for i in idxs]
                                        for kk in ['val','start','end']]
        mults = [get_vm0(*ap_out[i]) for i in idxs]
        cds  = [aps_mj(*ap_out[i][:2])['createdDateTime'] for i in idxs]
        chgs = [aps_mj(*ap_out[i][:2])['changes'] for i in idxs]
        
        # Check if the starts and ends 'line up'
        ssend_flg = check_ssend_flg(stts,ends)
        
        if len(set(vals))==1 and len(set(mults))==1:
            # First, if all values the same and length no ambiguity
            idx = 0
            vvmult.append(get_vm0(*ap_out[idxs[idx]]))
            vvals.append(get_data(*ap_out[idxs[idx]])['val'])
            vvx.append(0)
        elif ssend_flg:
            # If the starts & ends are contiguous, no ambiguity:
            vvl = contiguous_duplicates(stts,ends,vals,nomps[-1],dlo,dhi)
            vvmult.append(1)
            vvals.append(vvl)
            vvx.append(0)
        else:
            # Finally, if still no good, then find max / min outage rates
            mx, mn = minmax_reconciliation(stts,ends,vals,nomps[-1],dlo,dhi)
            vvmult.append(1)
            vvals.append(mx)
            vvx.append(mx-mn)
    
    return vvmult,vvals,nomps,vvx,mout


def block_process_aps(dstart,dend,kksD,APs,mm,kks,ei=True,):
    """Block process the APs for a range of dates at half-hourly timesteps.
    
    Inputs
    ---
    dstart, dend - start/end dates to process
    kks, kksD, APs, mm - see help(eomf.load_aps)
    ei - exclude_intermittent, help(eomf.process_aps)
    
    Returns
    ---
    - drange - datetimes
    - dpsX - the p/f/t aggregated outages
    - dpsXx - the p/f/t additional potential outages
    - moX - the p/f/t individual generator outages
    - moXx - the p/f/t additional potential individual generator outage
    """
    # Build the clocks
    drange = np.arange(dstart,dend,timedelta(0,3600),dtype=object)
    dr_pair = np.c_[drange,np.r_[drange[1:],dend]]
    
    # Get forced and planned outages,
    APsP = {kk:{k:[v for v in d if v['businessType']=='A53'] 
                        for k,d in dd.items()} for kk,dd in APs.items()}
    APsF = {kk:{k:[v for v in d if v['businessType']=='A54'] 
                        for k,d in dd.items()} for kk,dd in APs.items()}
    APsX = {'p':APsP,'f':APsF,'t':APs}
    pft = ['p','f','t',]
    
    with Bar('Process APs', suffix='%(percent).1f%% - %(eta)ds',
                                                    max=len(dr_pair)) as bar:
        dpsX,dpsXx = [{k:np.zeros(len(drange)) for k in pft} for i in range(2)]
        moX, moXx = [{k:mtDict(mm,) for k in pft} for i in range(2)]
        
        for i,(dlo,dhi) in enumerate(dr_pair):
            # First check there is data for the period
            clk_i = np.abs((dlo-kksD)//timedelta(1))
            if sum(clk_i==0)<1:
                if sum((clk_i==1))==2:
                    print(f'\nAmbiguous datetime {dlo} (i = {i}). Using earlier date.\n')
                else:
                    print(f'No data for {dlo} (i = {i}). leaving as zero.\n')
                    continue
            
            # If there is, pull out the data for that date.
            isel_kk = np.argmin(clk_i)
            for fpt in pft:
                vvmult,vvals,nomps,vvx,mout = \
                        process_aps(APsX[fpt][kks[isel_kk]],dlo,dhi,mm,ei,)
                
                # Update the dict of generator outages
                dout = npa(vvmult)*(npa(nomps) - npa(vvals))
                _ = [moX[fpt][m].append([i,d]) for m,d in zip(mout,dout)]
                _ = [moXx[fpt][m].append([i,vx]) for m,vx in zip(mout,vvx)
                                                                  if vx!=0]
                
                # Calculate the output
                dpsX[fpt][i] = sum(dout)
                dpsXx[fpt][i] = sum(vvx) # vvmult=1 for times by construction
            
            bar.next()
    
    return drange, dpsX, dpsXx, moX, moXx

# Force the dict order
genXmlData = odict({
        'mrid':['registeredResource.mRID',],
        'name':['registeredResource.name'],
        'gType':['MktPSRType','psrType'], # generation type
        'cap':['Period','Point','quantity',],
        'unit':['quantity_Measure_Unit.name',],
        'tStt':['Period','timeInterval','start',],
        'tEnd':['Period','timeInterval','end',],
            })

def find_recur(ts,flist):
    """Recursively go through xml element ts using 'find' on flist list."""
    if len(flist)==1:
        # If final element, return data
        return ts.find(flist[0]).text
    else:
        # Otherwise call again
        return find_recur(ts.find(flist[0]),flist[1:])

def process_icppu(r):
    """Process the ICPPU data to create a table for a given year."""
    root = ETnse(ET.XML(r.content.decode('utf-8')))
    data = []
    tss = root.findall('TimeSeries')
    for ts in root.findall('TimeSeries'):
        data.append([find_recur(ts,v) for v in genXmlData.values()])
    
    head = list(genXmlData.keys())
    return head,data

# getOutageUrl and getOutageData for entsoeOutageDownload
def getOutageUrl(code,d0,d1,offset_flag_i,):
    """Get the url for outage data from d0 to d1.
    
    If offset_flag_i is None, then no offset, else offset of 200*offset_flag_i
    """
    if offset_flag_i is None:
        url = getUrl('unvblGp',code,2018,opts=None)
    else:
        url = getUrl('unvblGp',code,2018,opts=[f'offset={200*offset_flag_i}'])
    
    url = url.replace('2018010100',d2s(d0),)
    url = url.replace('2019010100',d2s(d1),)
    return url

def getOutageData(code,d,dT,N=1,allow_retry=True,):
    """Get the dates, urls, results and status codes for day d and N splits."""
    urls, rs, scs = mtList(3)
    for i in range(N):
        offset_flag_i = i if N>1 else None
        urls.append(getOutageUrl( code,d,d+dT,offset_flag_i ))
        rs.append(reqData(urls[i]))
        scs.append(rs[i].status_code)
    
    # Have one timeout retry
    if any([sc==401 for sc in scs]) and allow_retry:
        print('401 returned, try again in 5 seconds.')
        time.sleep(5)
        urls,rs,scs = getOutageData(code,d,dT,N,allow_retry=False)
    
    return urls,rs,scs

# FUNCTIONS for entsoeProductionDownload.py
# First get the data common to all units
common_set = odict({
            'bz':['biddingZone_Domain.mRID'],
            'impl_date':['implementation_DateAndOrTime.date'],
            'rr_mrid':['registeredResource.mRID'],
            'rr_name':['registeredResource.name'],
            'rr_loc':['registeredResource.location.name'],
            'rr_psrType':['MktPSRType','psrType',],
            })

unit_set = odict({
            'mRID':['mRID'],
            'name':['name'],
            'loc':['generatingUnit_Location.name'],
            'psrType':['generatingUnit_PSRType.psrType'],
            'nomP':['nominalP'],
            # 'unit':'nominalP', # is an attrib rather than text
            })

unit_head =  list(unit_set.keys()) + ['unit'] + list(common_set.keys())

def process_production(r):
    root = ETnse(ET.XML(r.content.decode('utf-8')))

    # The data has two layers, the PowerSystemResource and the individual
    # GeneratingUnit_PowerSystemResources generators.
    tss = root.findall('TimeSeries')
    data = []
    for ts in root.findall('TimeSeries'):
        data_first = {k:find_recur(ts,v) for k,v in common_set.items()}
        # Then go through each unit and find the powers etc
        units = ts.find('MktPSRType').findall(
                                    'GeneratingUnit_PowerSystemResources')
        for unit in units:
            data_unit = {k:find_recur(unit,v) for k,v in unit_set.items()}
            data_unit['unit']  = unit.find('nominalP').attrib['unit']
            data_unit.update(**data_first)
            data.append([data_unit[h] for h in unit_head])
    
    return unit_head, data

def get_nov_day(yr):
    """Get the day of the month of the first Sunday of November.
    
    Based on method of the same name from dataClasses.aSys.
    """
    dayOfWeek = 6 - datetime(yr,11,1).weekday() # start on sunday
    return 1+dayOfWeek


# Table 3, 4, 5 from NGESO EMR ECR 2020. DATA as ECR 2019, T-1.
# NB: names are slightly different from, aSys.avlbty
avlbty_ecr = {
        'steam_oil':0.9126,
        'ocgt':0.9498,
        'recip':0.9498,
        'nuclear':0.8122,
        'hydro':0.8965,
        'ccgt':0.9000,
        'chp':0.9000,
        'coal':0.8581,
        'biomass':0.8581,
        'waste':0.8581,
        'dsr':0.8614,
        'wind_onshore':0.0781,
        'wind_offshore':0.1113,
        'solar':0.0234,
        'ess_05':0.1226,
        'ess_10':0.2470,
        'ess_15':0.3696,
        'ess_20':0.4866,
        'ess_25':0.5868,
        'ess_30':0.6593,
        'ess_35':0.7038,
        'ess_40':0.7298,
        'ess_45':0.7503,
        'ess_50':0.9508,
        'ess_55':0.9508,
        None:np.nan,
        }

cnvtr_ecr = {
        'Biomass':'biomass',
        'Fossil Brown coal/Lignite':'coal',
        'Fossil Coal-derived gas':'ccgt',
        'Fossil Gas':'ccgt',
        'Fossil Hard coal':'coal',
        'Fossil Oil':'steam_oil',
        'Fossil Oil shale':'steam_oil',
        'Fossil Peat':'coal',
        'Geothermal':None,
        'Hydro Pumped Storage':'hydro',
        'Hydro Run-of-river and poundage':'hydro',
        'Hydro Water Reservoir':'hydro',
        'Marine':'hydro',
        'Nuclear':'nuclear',
        'Other':'chp',
        # 'Other':None,
        'Other renewable':None,
        # 'Other renewable':None,
        'Solar':'solar',
        'Waste':'waste',
        'Wind Offshore':'wind_offshore',
        'Wind Onshore':'wind_onshore',
        }

# Table 4, BSC Loss of Load Probability Calculation Statement Version 2, 2019
# https://www.elexon.co.uk/documents/bsc-codes/lolp/loss-of-load-probability-calculation-statement/
# Accessed 10/10/21
# Availability factors from one-hour forecast MEL
avlbty_elexon = {
    'oil':0.998,
    'ocgt':0.997,
    'nuclear':0.998,
    'hydro':0.988,
    'pumped_storage':0.998,
    'ccgt':0.989,
    'coal':0.986,
    None:np.nan,
    }

cnvtr_elexon = {
        'Biomass':'coal',
        'Fossil Brown coal/Lignite':'coal',
        'Fossil Coal-derived gas':'ccgt',
        'Fossil Gas':'ccgt',
        'Fossil Hard coal':'coal',
        'Fossil Oil':'oil',
        'Fossil Oil shale':'oil',
        'Fossil Peat':'coal',
        'Geothermal':None,
        'Hydro Pumped Storage':'hydro',
        'Hydro Run-of-river and poundage':'hydro',
        'Hydro Water Reservoir':'hydro',
        'Marine':'hydro',
        'Nuclear':'nuclear',
        'Other':'ccgt',
        # 'Other':None,
        'Other renewable':None,
        # 'Other renewable':None,
        'Solar':'solar',
        'Waste':'coal',
        'Wind Offshore':'wind_offshore',
        'Wind Onshore':'wind_onshore',
        }

avlbty_flat = {'na':1.0}

flat_out = ['Wind Offshore','Wind Onshore','Solar','Other renewable',
                    'Geothermal',]
cnvtr_flat = {k:'na' for k in cnvtr_elexon.keys() if k not in flat_out}
for k in flat_out:
    cnvtr_flat[k] = None


# date to string functions. NB: assumes ALL are in UTC.
d2s = lambda d: d.isoformat()[:13].replace('-','').replace('T','')
s2d = lambda s: datetime(*[int(v) for v in [s[:4],s[4:6],s[6:8],s[8:]]])
m2s = lambda m: '-'.join([f'{ii:02d}' for ii in [m.year,m.month,m.day]])
s2m = lambda s: datetime(*[int(v) for v in s.split('-')])

def get_utc_kk(kk,cc):
    """Convert a local datetime string to UTC for country cc."""
    cc = cc if cc!='I0' else 'IE'
    
    # Based on s2d
    dt_str = [int(v) for v in [kk[:4],kk[4:6],kk[6:8],kk[8:]]]
    
    # Update to UTC
    tt_ = datetime(*dt_str,tzinfo=gettz(country_timezones[cc][0]) )
    tt = datetime(*dt_str,) - tt_.utcoffset()
    return d2s(tt)

# ============

class bzOutageGenerator():
    """Class for building and validating outages for peak seasons.
    
    Assumes independent outages for generators; technologies are modelled using
    geometric distributions for outage and repair times.
    
    """
    # Map from electricity capacity report titles to equivalent Elexon types
    cnvtr_ecr2elexon = {
        'steam_oil':'oil',
        'ocgt':'ocgt',
        'recip':'oil',
        'nuclear':'nuclear',
        'hydro':'hydro',
        'ccgt':'ccgt',
        'chp':'ccgt',
        'coal':'coal',
        'biomass':'coal',
        'waste':'coal',
        None:None,
        'solar':None,
        'wind_offshore':None,
        'wind_onshore':None,
    }
    
    def __init__(self,av_model='ecr',fleet_mode='all',):
        """Initialise bzOutageGenerator class.
        
        av_model options:
        - 'elexon', based on the LOLP methodology Table 4;
        - 'ecr', based on the monthly mean values from the ECRs;
        - 'flat', all values are trivially unity;
        - None, uses 'elexon'.
        
        fleet_mode options:
        - 'all', 'cc' for using all generators, or only those from that country
            for building the generator fleets self.fleets.
        
        """
        self.set_avlbty(av_model,)
        self.set_trn_avl()
        self.getGeneratorPortfolios(fleet_mode,)
    
    def build_unavl_model(self,nt=24*7*20*1,flt=None,yr=2020,
                                                assign=True,seed=None,):
        """Build the unavailability matrices for all countries in self.fleets.
        
        Inputs
        ---
        nt - length of data to be generated, in hours (default 1 20 week winter)
        flt - if None, use self.fleet
        yr - the fleet year to choose
        assign - if True (default) set to self.unavl
        seed - seed to pass to self.build_unavl_matrix
        
        Returns
        ---
        If assign=False, unavl is a Bunch of unavailabilities for each cc.
        
        """
        flt = self.fleets if flt is None else flt
        
        unavl = Bunch({'ccs':deepcopy( list(flt.keys()) )})
        for cc in unavl.ccs:
            fleet_yr = flt[cc] if yr is None else flt[cc][yr]
            unavl[cc] = {
            'v':np.concatenate([v for v in fleet_yr.values()]),
            'v_lt':np.concatenate([v*self.avlbty[self.cnvtr[k]]
                                for k,v in fleet_yr.items()]),
            }
            
            # Build the vector of generator sizes
            ng_all = len(unavl[cc]['v'])
            unavl[cc]['ua'] = np.zeros((ng_all,nt),dtype=bool)
            
            # AA = [] # <-- alternative method
            i0 = 0
            for k,v in fleet_yr.items():
                ng = len(v)
                k_ = self.cnvtr[k]
                k__ = self.cnvtr_ecr2elexon[k_]
                unavl[cc]['ua'][i0:i0+ng] = self.build_unavl_matrix(
                        self.trn_avl[k__],self.avlbty[k_],ng,nt,seed=seed,)
                i0+=ng
                # <-- alternative method
                # AA.append(self.build_unavl_matrix(
                            # self.trn_avl[k__],self.avlbty[k_],ng,nt))
            
            # Also set the index of generators that have non-zero unavailability
            unavl[cc]['isel'] = np.where(
                        np.not_equal(np.nansum(unavl[cc]['ua'],axis=1),0))[0]
        
        if assign:
            self.unavl = unavl
        else:
            return unavl
    
    @staticmethod
    def getWinters(dataIn,yrStt=2014,nYr=6,mode='f',vrbl=None):
        """ Get winter data for the structure passed in.
        
        Returns a sctXt object with a new timeseries.
        
        'dataIn' should either be:
        - an sctXt object, or
        - an object with a '__getitem__', and 'vrbl' should be passed in.
        
        'mode' can be either:
        - 'f' [full]: all winter days (20 weeks from 1st Sunday of Nov)
        - 'X' [without Xmas]: without 14 days of Christmas
        - 'A' [all]: start on 1st Jan and go for nYr years
        - 'djf'/'mam',/'jja'/'son': month triplets. 
                                    NB: djf goes from current to next yr.
        
        """
        # Get the 1st Sunday of Nov and 20 weeks after that
        dtsStt = []
        dtsEnd = []
        for i in range(nYr):
            novDate = get_nov_day(yrStt+i)
            dtsStt.append( datetime(yrStt+i,11,novDate) )
            dtsEnd.append( dtsStt[-1] + timedelta(7*20) )

        if mode=='X':
            # Get the Christmas dates (following Wilson PMAPS 2018)
            xmasStt = []
            xmasEnd = []
            for i in range(nYr):
                if dtsStt[i].day==7:
                    dWk = 6
                else:
                    dWk = 7
                xmasStt.append(dtsStt[i] + timedelta(dWk*7))
                xmasEnd.append(dtsStt[i] + timedelta((dWk+2)*7))
        
        
        # Stick these together into an array and find the indexes
        if mode=='f':
            dates = np.array([dtsStt,dtsEnd]).T
        elif mode=='X':
            dates = np.array([dtsStt,xmasStt,xmasEnd,dtsEnd]).T
        elif mode=='A':
            dates = np.array([[datetime(yrStt,1,1,)],
                             [datetime(yrStt+nYr,1,1,)]]).T
        elif mode in ['djf','mam','jja','son']:
            mos = {'djf':[12,3],'mam':[3,6],'jja':[6,9],'son':[9,12],}
            dY1 = 1 if mode=='djf' else 0
            dates = []
            for ii in range(nYr):
                dates.append([datetime(yrStt+ii,mos[mode][0],1),
                              datetime(yrStt+ii+dY1,mos[mode][1],1)])
            
            dates = np.array(dates)
        
        
        idxs = np.searchsorted(dataIn.t,dates)
        
        # Pull out the variables of interest between those index pairs.
        if vrbl is None:
            xx = dataIn.x
        else:
            xx = dataIn[vrbl]
        
        if xx.ndim==2:
            data = np.zeros((0,xx.shape[1],),dtype=float,)
        else:
            data = np.array([],dtype=float,)
        
        tms = np.array([],dtype=object)
        for idx_ in idxs:
            for i0,i1 in zip(idx_[::2],idx_[1::2]):
                data = np.r_[data,xx[i0:i1]]
                tms = np.r_[tms,dataIn.t[i0:i1]]
        
        return sctXt(x=data,t=tms)
    
    @staticmethod
    def build_unavl_matrix(trn_avl,lt_avl,ng,nt,seed=None,):
        """Build an unavailability matrix, with i,j = 1 if in outage.
         
        Uses the geometric function to simulate the number of transitions to 
        failure/repair.
        
        The markov chain generator used is based on the follow description:
        https://python.quantecon.org/finite_markov.html
        
        Inputs
        ---
        trn_avl - the one-transition availbility
        lt_avl - the long-term availability
        ng - number of generators
        nt - number of time periods to return
        seed - the random seed used to simulate the MC output; None for random
        
        Returns
        ---
        unavl - ng x nt matrix with 1 in unavailable.
        """
        # Random number generation initialisation
        rng_ = rngSeed(seed=seed,)
        seed = rng.integers(0,2**32)
        
        # If in trn_avl or 
        if ng==0 or np.isnan(trn_avl):
            return np.zeros((ng,nt,))
        
        # Calculate the transition probabilities
        lmd = 1-trn_avl
        mu = lmd*lt_avl/(1-lt_avl)
        
        # Build the Markov Chaing probability matrix and initial state
        PP = [[trn_avl,1-trn_avl],[mu,1-mu]]
        state_init = rng_.choice([0,1],size=(ng,),p=[lt_avl,1-lt_avl])
        
        # Simulate the outages
        mc = qe.MarkovChain(PP)
        unavl = mc.simulate(ts_length=nt,init=state_init,random_state=seed,)
        
        return unavl
    
    @staticmethod
    def clean_inPrdData(data):
        """Clean the installed production data matrix downloaded from ENTSOe.
        
        """
        # First strip the first column
        data = [r[1:] for r in data]
        
        # Then go through and replace empty columns with NaNs
        dc = np.zeros((len(data),len(data[0])))
        for i,r in enumerate(data):
            for j,v in enumerate(r):
                # n/e, N/A as not expecting, not applicable
                dc[i,j] = np.nan if v in ['n/e','N/A'] else float(v)
        return dc
    
    def set_avlbty(self,av_model=None,):
        """Set the availability model. See help(self.__init__) for options."""
        if av_model=='elexon':
            self.avlbty = avlbty_elexon
            self.cnvtr = cnvtr_elexon
        elif av_model in ['ecr',None]:
            self.avlbty = avlbty_ecr
            self.cnvtr = cnvtr_ecr
        elif av_model=='flat':
            self.avlbty = avlbty_flat
            self.cnvtr = cnvtr_flat
        
        # Get the availability of conventional-only generators
        not_conv = [ # Elexon   ECR
            'Wind Onshore',     'wind_onshore',
            'Wind Offshore',    'wind_offshore',
            'Solar',            'solar',
            'Hydro Run-of-river and poundage', # lumped all as normal hydro
        ]   
        self.avlbty_conv = deepcopy(self.avlbty)
        _ = [self.avlbty_conv.__setitem__(k,0) for k in self.avlbty
                                                        if k in not_conv]
    
    def set_trn_avl(self,):
        """Set the one-hour transition probability."""
        self.trn_avl = avlbty_elexon
    
    def getGeneratorPortfolios(self,fleet_mode,):
        """Get the generator portfolios for all countries using entsoe.
        
        For most of these, self._.ccs lists the countries present for 
        iterating over.
        
        Inputs
        ---
        fleet_mode - gets passed into getGenFleets for each country.
        
        Sets
        ---
        - self.nseInPrd, a Bunch with entsoe total generation data,
        - self.nsePng, the entsoe production & generation countries,
        - self.fleets, the fleet, built using the previous two datasets.
        
        """
        listEq = lambda ll: all([ll[0]==l for l in ll])
        
        # First load the totals for each balancing zone
        dn = os.path.join(fn_root,'misc','inpr',)
        
        hs = []; rs = []
        self.nseInPrd = Bunch()
        self.nseInPrd.ccs = []
        for fn in os.listdir(dn):
            cc = fn.split('_')[0]
            head,data = csvIn(os.path.join(dn,fn))
            hs.append(head)
            rs.append([d[0] for d in data])
            self.nseInPrd[cc] = bzOutageGenerator.clean_inPrdData(data)
            self.nseInPrd['ccs'].append(cc)
        
        # Norway only goes to 2020 for some reason - assume similar to 2020
        for cc in [c for c in self.nseInPrd.ccs if 'NO-' in c]:
            self.nseInPrd[cc].__setitem__((slice(None),slice(-1,None)),
                                o2o(self.nseInPrd[cc][:,-3]))
        
        if not (listEq(hs) and listEq(rs)):
            raise Exception('List row headings and column headings not equal!')
        
        self.nseInPrd.h = [int(h[:4]) for h in hs[0][1:]]
        self.nseInPrd.r = rs[0]
        
        # Then, for countries which are split, also add the 'totals'
        cc_update = set([cc[:2] for cc in self.nseInPrd.ccs 
                                    if cc[:2] not in self.nseInPrd.ccs])
        
        # sum list add concatante
        list_add = lambda lxx: np.nansum(np.stack(lxx),axis=0)
        
        for cc in cc_update:
            self.nseInPrd[cc] = list_add([self.nseInPrd[cc_] 
                            for cc_ in self.nseInPrd.ccs if cc in cc_])
            self.nseInPrd['ccs'].append(cc)
        
        # Then load the production and generation, based on fig_pngUnits
        # NB: we only use the Dec 2020 data for now as it seems units are
        # not taken off.
        self.nsePng = Bunch()
        self.fleets = Bunch()
        pngHeads = odict({
                       'mRID':str,
                       'name':str,
                       'nomP':float,
                       'impl_date':s2m,
                       })
        self.nsePng.head = list(pngHeads.keys())
        
        dpng = os.path.join(fn_root,'entsoeData','png',)
        self.nsePng.ccs = get_path_dirs(dpng,'names',)
        for cc in self.nsePng.ccs:
            self.nsePng[cc] = {}
            fns = [fn for fn in get_path_files(os.path.join(dpng,cc))
                                            if '2020-12-01.csv' in fn]
            heads, data = mtList(2)
            update_hd = lambda dd: [heads.append(dd[0]),data.extend(dd[1])]
            _ = [update_hd(csvIn(fn)) for fn in fns]
            assert(all([heads[0]==h for h in heads]))
            
            for kk,pmap in PSRTYPE_MAPPINGS.items():
                d_pmap = [d for d in data if (d[heads[0].index('psrType')]==kk)]
                self.nsePng[cc][pmap] = [[vf(r[heads[0].index(k)]) 
                               for k,vf in pngHeads.items()] for r in d_pmap]
        
        # Set the list of the portfolio of all generators
        self.getAllGenList()
        
        for cc in self.nsePng.ccs:
            self.fleets[cc] = self.getGenFleets(cc,mode=fleet_mode,)
    
    @staticmethod
    def pwr2fleet(ppwrs,val):
        """Convert list of powers ppwrs to a fleet of size val."""
        ppwrs_mul = int(np.ceil(val/sum(ppwrs)))
        ppwrs_list = ppwrs*ppwrs_mul
        isel = np.argmax(np.cumsum(ppwrs_list)>val)
        gen_sub = val - sum(ppwrs_list[:isel])
        return np.r_[ppwrs_list[:isel],gen_sub]
    
    def getAllGenList(self,):
        """Set self.allGenList as the list of all generators from all countries.
        
        Uses self.nsePng for this.
        """
        self.allGenList = Bunch({'ccs':self.nsePng.ccs})
        ipwr = self.nsePng.head.index('nomP')
        for vv in self.nseInPrd.r[:-1]:
            # Get the generator fleets - weirdly, there is no lignite?
            vvpng = 'Fossil Hard coal' if vv=='Fossil Brown coal/Lignite' \
                                                                    else vv
            self.allGenList[vv] = [r[ipwr] for cc in self.nsePng.ccs for 
                                            r in self.nsePng[cc][vvpng]]
            if len(self.allGenList[vv])==0:
                print(f'Warning - no data for scnFleets for {vv}!')
    
    def getGenFleets(self,cc,vbs=False,mode='all',):
        """Get the generator fleets using the LILO method for system cc.
        
        Mode: if 'cc', only use the country generators for LILO; otherwise, if
        'all' then use all generators.
        """
        
        # Get the nseInPrd table andlist of generators
        cc = 'IE' if cc=='I0' else cc # use same data for Irish codes
        cc_tbl = self.nseInPrd[cc]
        rr = self.nseInPrd.r[:-1] # ignore Total Grand Capacity
        gens = self.nsePng[cc] if mode=='cc' else self.allGenList
        
        # indexes and dates
        idate,ipwr = [self.nsePng.head.index(vv) 
                                            for vv in ['impl_date','nomP']]
        
        fleets = {yr:mtOdict(rr) for yr in self.nseInPrd.h}
        for j,yr in enumerate(self.nseInPrd.h):
            for i,r in enumerate(rr):
                if cc_tbl[i,j]>0 and len(gens[r])>0:
                    if mode=='cc':
                        # evaluate if gens existed at the start of the yr
                        igensel = [i for i,g in enumerate(gens[r]) 
                                                    if g[idate].year<yr]
                        igenout = [i for i in range(len(gens[r])) 
                                                    if i not in igensel]
                        gensr_ = np.array(gens[r])
                        
                        # get backwards/forward date orders for igensel/igenout
                        isrt = np.argsort([gensr_[ii,idate] 
                                                for ii in igensel])[::-1] 
                        isrt_out = np.argsort([gensr_[ii,idate] 
                                                        for ii in igenout])
                        
                        pwrs = gensr_[igensel][isrt][:,ipwr].astype(float)
                        pwrs_out=gensr_[igenout][isrt_out][:,ipwr].astype(float)
                        pwrs_all = np.r_[pwrs,pwrs_out]
                    elif mode=='all':
                        pwrs = self.allGenList[r]
                        pwrs_all = pwrs
                    
                    if sum(pwrs)>cc_tbl[i,j]:
                        # If there are sufficient generators, 
                        # then pick the most recent:
                        fleets[yr][r] = self.pwr2fleet(pwrs,cc_tbl[i,j])
                    elif sum(pwrs_all)>cc_tbl[i,j]:
                        # If there are not but the total works, then go forward
                        fleets[yr][r] = self.pwr2fleet(pwrs_all,cc_tbl[i,j])
                    elif sum(pwrs_all)!=0:
                        nrot = int(cc_tbl[i,j]//sum(pwrs_all) + 1)
                        pwrs_aug = np.concatenate([pwrs_all]*nrot)
                        fleets[yr][r] = self.pwr2fleet(pwrs_aug,cc_tbl[i,j])
                    else:
                        if vbs:
                            print(sum(pwrs_all),yr,r)
                        
                        raise Exception('not yet implemented')
                elif cc_tbl[i,j]>0 and len(gens[r])==0:
                    if vbs:
                        print(
                           f'No data for {cc}, {yr}, {r}, ({cc_tbl[i,j,]} MW)')
                    fleets[yr][r] = np.array([cc_tbl[i,j]])
        _ = [[ff.__setitem__(k,np.zeros((0,),)) for k,v in ff.items()
                            if type(v) is list] for yr,ff in fleets.items()]
        
        return fleets


