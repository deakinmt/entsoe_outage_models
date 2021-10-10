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
from mosek.fusion import *

from eom_utils import repl_list, fn_root, data2csv, structDict, sctXt,\
        tDict2stamps, boxplotqntls, nanlt, nangt, plotEdf, cdf2qntls, rerr
exec(repl_list)
from entsoe_py_data import BIDDING_ZONES, PSRTYPE_MAPPINGS
from progress.bar import Bar

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
                    'DK-1','DK-2','NL','FR','BE','IE','GB','I0','DE',]
    
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
    """
    xmlns = (root.tag).split('}')[0]+'}'
    timeSeries = root.findall(xmlns+'TimeSeries')
    
    if len(reqType.split('_'))>1:
        reqT0,bt = reqType.split('_BT')
        timeSeries = [ts for ts in timeSeries \
                            if ts.find(xmlns+'businessType').text==bt]
    
    crvs = []
    tss = []
    timeRes0 = timeSeries[0].find(xmlns+'Period').find(xmlns+'resolution')
    if timeRes0.text[-1]=='M':
        res = int(''.join(filter(str.isdigit,timeRes0.text)))
        dtHr = timedelta(0,res*60) # assume this stays constant
    elif timeRes0.text[-1]=='D':
        res = int(''.join(filter(str.isdigit,timeRes0.text)))
        dtHr = timedelta(1,) # assume this stays constant
    
    unit0 = timeSeries[0].find(xmlns+reqInfo[reqType]['unitHead']).text
    
    for ts in timeSeries:
        prd = ts.find(xmlns+'Period')
        # first check the time resolution +  is ok.
        timeRes = prd.find(xmlns+'resolution')
        if timeRes0.text!=timeRes.text:
            print('Time before: {}, time after:{}'.format(
                                            timeRes0.text,timeRes.text))
            raise ValueError('The time res. changes! Code needs updating')
        unit = ts.find(xmlns+reqInfo[reqType]['unitHead']).text
        if unit0!=unit:
            print('Unit before: {}, unit after:{}'.format(
                                            unit0,unit))
            raise ValueError('The time res. changes! Code needs updating')
        
        
        tis = prd.find(xmlns+'timeInterval')
        ti0 = tis.find(xmlns+'start').text
        if ti0[-1]!='Z':
            print('Warning! Datetime is NOT utc.')
        dt0 = datetime.fromisoformat(ti0[:-1])
        pts = prd.findall(xmlns+'Point')
        for pt in prd.findall(xmlns+'Point'):
            ii = int(pt.find(xmlns+'position').text) - 1
            crvs.append(float(
                          pt.find(xmlns+reqInfo[reqType]['dataName']).text))
            tss.append(dt0+ii*dtHr)
    
    nT = int((max(tss)-min(tss))/dtHr)
    mT = min(tss)
    tssStatic = np.array([mT+i*dtHr for i in range(nT+1)])
    curveStatic = np.nan*np.zeros(len(tssStatic))
    for i,t in enumerate(tss):
        curveStatic[int((t - tssStatic[0])/dtHr)]=crvs[i]
    return (tssStatic,curveStatic),unit0

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
    return all([e<=s for s,e in zip(stts[1:],ends[:-1])])

def contiguous_duplicates(stts,ends,vals,nomps,dlo,dhi,):
    """Create a contiguous equivalent when there is a stepped curve.
    
    Inputs
    ---
    stts, ends - start & end times
    vals - 
    nomps - the nominal power of the device
    
    Returns
    ---
    vvm - set equal to one, as all info captured in vvl
    vvl - 
    
    """
    if not check_ssend_flg(stts,ends):
        print('A clash in the non-unique APs in contiguous duplicates!')
        return np.nan, np.nan
    
    dt = dhi - dlo
    
    # The initial dd, pp
    dd = [(dlo,max(dlo,stts[0]))]
    pp = [nomps[-1]]
    
    # Add the first point manually
    dd.append((max(dlo,stts[0]),ends[0]))
    pp.append(vals[0])
    dd.append((ends[0],stts[1]))
    pp.append(nomps[-1])
    
    # Add the intermediate points
    for i in range(1,len(stts)-1):
        pp.append(vals[i])
        dd.append((stts[i],ends[i]))
        pp.append(nomps[-1])
        dd.append((ends[i],stts[i+1]))
    
    # Add the final point manually
    pp.append(vals[-1])
    dd.append((stts[-1],min(ends[-1],dhi)))
    pp.append(nomps[-1])
    dd.append((min(ends[-1],dhi),dhi))
    
    return 1.0, (o2o(np.diff(dd))/dt).dot(pp)

def load_aps(cc,sd,rerun=True,save=False,):
    """Load the availability periods list APs for country cc.
    
    Is quite slow unfortunately.
    
    WARNING - at the moment probably WONT work with DK-1 (etc)
    
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
    print(f'Loading {cc} APs...')
    ap2clk = lambda ap,ss: datetime.fromisoformat(
            ap.find('timeInterval').find(ss).text.replace('Z',':00'))
    
    # Approach: a day has a list of mRIDs. If a mRID does not have it's key
    # information in mRIDdict, then this is added. Then, for each availability
    # period, the data is saved.
    mm = {} # mRIDmaster dict
    
    APs = {} # each file
    for fn in get_path_files(os.path.join(ld_,cc)):
        kk = fn.split('_')[-3]
        
        # If not existing yet, add new one.
        if kk not in APs.keys():
            APs[kk] = {}
        
        with zipfile.ZipFile(fn, 'r') as zip_ref:
            for zr in zip_ref.infolist():
                root = ET.XML(zip_ref.read(zr))
                rr = ETnse(root)
                ts = rr.find('TimeSeries') # only ever one (it seems for now!)
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
    print(f'APs for {cc} loaded.')
    
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

def load_dps(dstart,dend,cc,sd,rerun=True,save=False,):
    """Block process the APs to find the dps values for each country.
    
    Inputs: see help(eomf.load_aps) for inputs
    
    Returns
    ---
    drange - the clock
    dpsF - the forced outages (nominally all)
    dpsP - the planned outages (nominally zero)
    """
    sd_ = os.path.join(sd,'output_cache','DPs',)
    _ = os.mkdir(sd_) if not os.path.exists(sd_) else None
    fn_ = os.path.join(sd_,f'{cc}_APs{d2s(dstart)}-{d2s(dend)}.pkl')
    
    # If not rerun, reload the cached data and return
    if not rerun:
        with open(fn_,'rb') as file:
            data = pickle.load(file)
        print(data['readme'])
        return data['drange'], data['dpsF'], data['dpsP']
    
    t0 = timer()
    print(f'========== Processing {cc} APs...')
    APs, mm, kks, kksD = load_aps(cc,sd,rerun=False,save=False)
    drange,dpsF,dpsP = block_process_aps(dstart,dend,kksD,APs,mm,kks)
    
    if save:
        nl = '\n'
        readme = f'{cc} DPS created at {datetime.today().isoformat()}{nl}'\
                + f'- Start date: {dstart}{nl}'\
                + f'- End date: {dend}{nl}'\
                + f'- Time taken to build: {timer() - t0:1g}s{nl}'
        
        with open(fn_,'wb') as file:
            pickle.dump({'drange':drange,'dpsF':dpsF,'dpsP':dpsP,
                                                'readme':readme},file)
    
    return drange,dpsF,dpsP

def process_aps(APsK,dlo,dhi,mm):
    """Use APs to find the forced and unforced outages.
    
    Inputs
    ---
    APsK - availabilities dict value that will be used from dlo to dhi
    dlo - the start time (inclusive)
    dhi - the end time (exclusive)
    mm - the dict of generator data
    
    Outputs [I think these are not independent...?]
    ---
    vvmult - a list of multipliers (1 if outage during whole period)
    vvals - the value reported in the entsoe files
    nomps - the corresponding nominal power
    
    """
    # dhi = dlo+timedelta(0,1800) if dhi is None else dhi
    mmRdr = list(APsK.keys())

    # First, pull out all of the possibilities:
    ap_out = []
    for i,m in enumerate(mmRdr):
        for j,aps in enumerate(APsK[m]):
            if aps['docStatus']!='A09':
                for k,dd in enumerate(aps['data']):
                    if (dhi>dd['start'] and dlo<dd['end']):
                        ap_out.append([m,j,k,])
    
    # m,j,k: mRID, 
    get_data = lambda m,j,k: APsK[m][j]['data'][k]
    get_nomP = lambda m,j,k: APsK[m][j]
    get_cd = lambda m,j,k: APsK[m][j]['createdDateTime']
    get_vm0 = lambda m,j,k: (get_data(m,j,k)['end'] - dlo)/(dhi - dlo)
    nom2float = lambda nomP: 0 if nomP is None else float(nomP)
    # nom2float = lambda nomP: np.nan if nomP is None else float(nomP)
    
    # Find unique vales and then duplicates
    mmsel = [a[0] for a in ap_out]
    unq_idxs = [i for i,m_ in enumerate(mmsel) if mmsel.count(m_)==1]
    mm_nunq = list(set([m_ for m_ in mmsel if mmsel.count(m_)!=1]))

    # Get the time multiplier, nom power, and AP value
    vvmult = [min(get_vm0(*ap_out[i]),1) for i in unq_idxs]
    vvals = [get_data(*ap_out[i])['val'] for i in unq_idxs]
    nomps = [nom2float(mm[mmsel[i]]['nomP']) for i in unq_idxs]

    # Deal with any duplicates
    for nunq in mm_nunq:
        idxs = [i for i,a_ in enumerate(ap_out) if a_[0]==nunq]
        
        chgs = [APsK[ap_out[i][0]][ap_out[i][1]]['changes'] for i in idxs]
        
        vals = [get_data(*ap_out[i])['val'] for i in idxs]
        stts = [get_data(*ap_out[i])['start'] for i in idxs]
        ends = [get_data(*ap_out[i])['end'] for i in idxs]
        ssend_flg = check_ssend_flg(stts,ends)
        
        mults = [min(get_vm0(*ap_out[i]),1) for i in idxs]
        cds = [get_cd(*ap_out[i]) for i in idxs]
        nomps.append(nom2float(mm[mmsel[idxs[0]]]['nomP']))
        
        if len(set(vals))==1 and len(set(mults))==1:
            # First, if all values the same and length no ambiguity
            idx = 0
            vvmult.append(min(get_vm0(*ap_out[idxs[idx]]),1))
            vvals.append(get_data(*ap_out[idxs[idx]])['val'])
        elif len(set(mults))==1:
            # Then, if the mults are all identical, choose most recent
            idx = cds.index(max(cds))
            vvmult.append(min(get_vm0(*ap_out[idxs[idx]]),1))
            vvals.append(get_data(*ap_out[idxs[idx]])['val'])
        elif len(set(cds))==1 and len(set(vals))==1:
            # Then, if there is only one date, combine to one
            vvmult.append(sum(mults))
            vvals.append(vals[0])
        elif ssend_flg:
            # If the starts & ends are contiguous, stitch together:
            vvm,vvl = contiguous_duplicates(stts,ends,vals,nomps,dlo,dhi)
            vvmult.append(vvm)
            vvals.append(vvl)
        elif not all(chgs) and any(chgs):
            # If mix of chgs and planned, try with continuous/just adding
            f = lambda xx: [x for x,c in zip(xx,chgs) if c]
            if sum(chgs)==1:
                vvm = min(get_vm0(*ap_out[f(idxs)[0]]),1)
                vvl = get_data(*ap_out[f(idxs)[0]])['val']
            else:
                vvm,vvl = contiguous_duplicates(f(stts),f(ends),f(vals),
                                                            nomps,dlo,dhi)
            vvmult.append(vvm)
            vvals.append(vvl)
        else:
            # Otherwise, complain!
            vvmult.append(np.nan)
            vvals.append(np.nan)
            print('A clash in the non-unique APs!')
            print(nunq,vals,len(set(cds)),dlo,)
            print(chgs)

    return vvmult,vvals,nomps

def block_process_aps(dstart,dend,kksD,APs,mm,kks):
    """Block process the APs for a range of dates at half-hourly timesteps.
    
    Inputs
    ---
    dstart, dend - start/end dates to process
    kks, kksD, APs, mm - see help(eomf.load_aps)
    
    Returns
    ---
    - drange - datetimes
    - dpsF - forced outages time series
    - dpsP - planned outages time series
    """
    drange = np.arange(dstart,dend,timedelta(0,3600),dtype=object)
    dr_pair = np.c_[drange,np.r_[drange[1:],dend]]
    
    # Get forced and planned outages,
    APsP = {kk:{k:[v for v in d if v['businessType']=='A53'] 
                        for k,d in dd.items()} for kk,dd in APs.items()}
    APsF = {kk:{k:[v for v in d if v['businessType']=='A54'] 
                        for k,d in dd.items()} for kk,dd in APs.items()}
    
    dpsF,dpsP = [np.zeros(len(drange)) for i in range(2)]
    for i,(dlo,dhi) in enumerate(dr_pair):
        isel_kk = np.argmin(np.abs((dlo-kksD)//timedelta(1)))
        for aps,dps in zip([APsF,APsP],[dpsF,dpsP]):
            vvmult,vvals,nomps = process_aps(aps[kks[isel_kk]],dlo,dhi,mm)
            dp = np.array(nomps) - np.array(vvals)
            dps[i] = sum(dp)
    
    return drange, dpsF, dpsP

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
avlbty_elexon = {
    'oil':0.998,
    'ocgt':0.997,
    'nuclear':0.998,
    'hydro':0.988,
    'pumped_storage':0.998,
    'ccgt':0.989,
    'coal':0.986,
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

# date to string functions
d2s = lambda d: d.isoformat()[:13].replace('-','').replace('T','')
s2d = lambda s: datetime(*[int(v) for v in [s[:4],s[4:6],s[6:8],s[8:]]])
m2s = lambda m: '-'.join([f'{ii:02d}' for ii in [m.year,m.month,m.day]])
s2m = lambda s: datetime(*[int(v) for v in s.split('-')])