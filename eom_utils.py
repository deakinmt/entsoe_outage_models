import sys, os, csv, socket, shutil, pickle, subprocess
from matplotlib import rcParams
import traceback
import matplotlib.pyplot as plt
from scipy import sparse
import numpy as np
from pprint import pprint
from datetime import datetime
import win32com.client
from datetime import datetime, date, timedelta
from dateutil import tz
from time import ctime
from bunch import *
from collections import OrderedDict as odict

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from numpy.random import MT19937, Generator, SeedSequence
from govuk_bank_holidays.bank_holidays import BankHolidays
from matplotlib import cm
from cmocean import cm as cmo
from hsluv import hsluv_to_rgb


# run exec(repl_import) to import a set of standard funcs from here
repl_list = """from eom_utils import es, csvIn, ppd, oss, tlps, saveFigFunc,\
 dds, set_day_label, openGallery, tl, mtDict, mtOdict, o2o, gDir, gdv, gdk,\
 ossf, ossm, get_path_files, get_path_dirs, mtList, sff, listT, ione, data2csv,\
 pound, og, basicTblSgn, saveDataFunc, npa
"""

pound = u'\u00A3'
npa = np.array

# A useful set of directories for saving/loading data from etc
fn_call = sys.argv[0].replace('/','\\').lower()
scriptCallingName = fn_call.split('\\')[-1].split('.')[0]
if '__file__' in dir():
    fd = os.path.dirname(__file__)
else:
    fd = os.path.dirname(fn_call)

fn_root = fd

es = "exec( open(r'"+fn_call+"').read() )"
gDir = os.path.join(fn_root,'gallery',scriptCallingName)


def csvIn(fn,hh=True,):
    """Uses the CSV object reader to read in a csv file.
    
    Assumes the first line is a header with the row names, the rest the 
    data of the file.
    
    See also: data2csv
    
    Inputs
    ---
    fn - the filename to read
    hh - if True, the return a head first.
    
    Returns
    ---
    head - the first line of the csv
    data - the rest of the lines of the data.
    
    """
    with open(fn,'r') as file:
        csvR = csv.reader(file)
        if hh:
            head = csvR.__next__()
        
        data = list(csvR)
    
    if hh:
        return head, data
    else:
        return data

def ppd(thing,fltr=None):
    """Pretty print dir(thing); with a simple first letter filter fltr."""
    pprint(letterFilter(dir(thing),fltr))

def oss(fn):
    """Convenience function, as os.startfile(fn)."""
    os.startfile(fn)

def tlps():
    """Convenience: plt.tight_layout(), plt.show()"""
    tl()
    plt.show()

def saveFigFunc(sd=gDir,**kwargs):
    """A simple wrapper function to save figures to sd.
    
    Always writes to a png file.
    
    If figname is not passed, the figure is named after the function that
    calls this function (as if by magic! ;) )
    
    If you want an EMF for inclusion with Word etc, the process is only
    semi-automated, as it seems that an inkscape conversion is required, which
    (at the moment) is called outside of python in svg_to_emf (see NCL 
    emails). This would be nice functionality to add in future though :)
    
    By default, the script saves a lo-res png to the gallery as well.
    
    Use 'cleanGallery' to clear all pngs in a gallery.
    ----
    kwargs
    ----
    figname: if specified, is the name of the figure saved to sd directory.
    dpi: dpi for png
    pdf: if True, also save as pdf
    svg: if True, also save as svg
    emf: if True, also save as an emf (saves both emf + svg)
    gOn: whether or not to save a (lo-res) png to the gallery folder
    pad_inches: inches to pad around fig, nominally 0.05
    sd_mod: if passed in, modifies the sd to sd/sd_mod

    """
    kwd = {
              'dpi':300,
              'pdf':False,
              'svg':False,
              'emf':False,
              'gOn':True,
              'pad_inches':0.05,
              'figname':None,
              'sd_mod':None,
              }
    kwd.update(kwargs)
    sd_mod = kwd['sd_mod']
    
    # create the gallery directory if not existing
    if not os.path.exists(gDir):
        os.mkdir(gDir)
        print('Created new gallery folder:', gDir)
    
    # For sd_mod, create file if not existing (first checks for sd)
    gDir_dn = gDir if sd_mod is None else os.path.join(gDir,sd_mod)

    if not sd_mod is None:
        if not os.path.exists(sd):
            raise Exception('Create initial sd first!')
        
        # Then create sd if it doesn't exist
        sd = sd if sd_mod is None else os.path.join(sd,sd_mod)
        _ = os.mkdir(sd) if not os.path.exists(sd) else None
        _ = os.mkdir(gDir_dn) if not os.path.exists(gDir_dn) else None

    # simple script that, by default, uses the name of the function that 
    # calls it to save to the file directory sd.
    print(whocalledme())
    if kwd['figname'] is None:
        kwd['figname'] = whocalledme(depth=3)
    
    fn = os.path.join(sd,kwd['figname'])
    print('\nSaved with saveFigFunc to\n ---->',fn)
    plt.savefig(fn+'.png',dpi=kwd['dpi'],pad_inches=kwd['pad_inches'])
    
    # Then save extra copies as specified.
    if kwd['pdf']:
        plt.savefig(fn+'.pdf',pad_inches=kwd['pad_inches'])
    if kwd['svg'] or kwd['emf']:
        plt.savefig(fn+'.svg',pad_inches=kwd['pad_inches'])
    if kwd['emf']:
        iscpPath = r'C:\Program Files\Inkscape\bin\inkscape.exe'
        subprocess.run([iscpPath,fn+'.svg','--export-filename',fn+'.emf',])
    if kwd['gOn']:
        fn_gallery = os.path.join(gDir_dn,kwd['figname'])
        if fn_gallery!=fn:
            plt.savefig(fn_gallery+'.png',dpi=100,pad_inches=kwd['pad_inches'])

def dds(data,n=5,*,cut=False,nan=False):
    """Data down-sample by n.
    
    data: dataset to downsample,
    n: number of downsample points,
    cut (keyword): remove datapoints at the end of data if not whole no. sets
    nan (keyword): calculated the nanmean rather than mean
    
    NB: if data is a time series and 'cut' is True, then the time variable t 
    needs to be t = [::n][:-1].
    """
    
    if (data.shape[0] % n)!=0:
        if not cut:
            raise ValueError('\n\nDDS data shape: {},\nDDS n = {},'.format(
                            *[data.shape,n])+'\n\t--> DDS failed!'+\
                              '\n\nUse cut=True opt to cut data.)')
        else:
            data = data.copy() # get rid of link to input data
            data = data[:-((len(data) % n))]
    if nan:
        # Not that well studied - updated 3/3/21
        x = []
        for i in range(len(data)//n):
            x.append(np.nanmean(data[i*n:(i+1)*n]))
        return x
    else:
        return sparse.kron( sparse.eye(len(data)//n),\
                                        np.ones(n) ).dot(data)/n

def set_day_label(hr=3,t=False,):
    """Convenience function for setting the x label/ticks for day plots.
    
    hr - number of hours to 'jump' in xticks
    t - 'tight' or not, if True then fit to (0,23), else (-0.3,23.3)
    """
    plt.xticks(np.arange(0,24,hr))
    if t:
        plt.xlim((0,23))
    else:
        plt.xlim((-0.3,23.3))
    
    plt.xlabel('Hour of the day')

def og():
    """Helper function, runs openGallery()."""
    openGallery()

def openGallery():
    """Equivalent to os.startfile(gDir)."""
    os.startfile(gDir)

def tl():
    """Convenience function, plt.tight_layout()"""
    plt.tight_layout()

def mtDict(keys):
    """Return a dict with all of keys set to empty lists."""
    mtDict = {}
    mtDict.update(zip(keys,mtList(len(keys))))
    return mtDict

def o2o(x,T=False):
    """If x is ndim=1, convert to ndim=2 as a vector; and vice versa.
    
    o2o as in 'one-to-one' vectors - is intended to allow statements such as
    X/y when y is only one dimension (and numpy then complains).
    
    If x.ndim==1:
    - T flag False [default] -> column vector
    - T flag True -> row vector.
    """
    if x.ndim==2:
        return x.flatten()
    elif x.ndim==1:
        if T:
            return x.reshape((1,-1))
        else:
            return x.reshape((-1,1))

def gdv(dd,n=0):
    """Get-dict-val; returns n-th val of dict dd."""
    return dd[list(dd.keys())[n]]

def gdk(dd,n=0):
    """Get-dict-key; returns n-th key of dict dd."""
    return list(dd.keys())[n]

def ossf(obj):
    """Convenience function, runs oss(obj.__file__)."""
    oss(obj.__file__)

def ossm(package):
    """Import [str] package, then open the location using ossf."""
    exec('import '+package)
    exec(f'ossf({package})')

def get_path_dirs(path,mode='paths'):
    """Get directories/folders in path (is reasonably reliable).
    
    Sometimes fails, e.g., with .git files, for whatever reason.
    
    See email notes for where this comes from.
    
    mode: either 
    - 'paths' (returns full path), or
    - 'names' (returns just the directory names)
    """
    if mode=='paths':
        return [f.path for f in os.scandir(path) if f.is_dir()]
    elif mode=='names':
        return [f.name for f in os.scandir(path) if f.is_dir()]

def get_path_files(path,mode='paths',ext=None):
    """Get the files in the folder path.
    
    Dual of get_path_dirs - see for doc.
    """
    if mode=='paths':
        lst = [f.path for f in os.scandir(path) if not f.is_dir()]
    elif mode=='names':
        lst = [f.name for f in os.scandir(path) if not f.is_dir()]
    
    if ext is None:
        return lst
    else:
        return [f for f in lst if f.split('.')[-1]==ext]

def mtList(N):
    return [ [] for _ in range(N)]

def sff(fn,tlFlag=True,**kwargs):
    """Convenience func to call saveFigFunc with fn as the figname arg.
    
    kwargs passed in go straight through to saveFigFunc; the exception is 
    tlFlag which calls tight_layout() to tidy things up; this can be disabled
    as necessary for some figures.
    """
    if tlFlag:
        tl()
    
    saveFigFunc(figname=fn,**kwargs,)

def listT(l):
    """Return the 'transpose' of a list.""" 
    return list(map(list, zip(*l)))

def ione(itrbl):
    """Return an iterable (list) with just the first element of itrbl.
    
    Useful for debugging for-loops with dicts etc.
    """
    return [itrbl.__iter__().__next__()]

def data2csv(fn,data,head=None):
    """Write list of lists 'data' to a csv.
    
    If head is passed, put as the first row.
    
    See also: csvIn.
    """
    if not head is None:
        data = [head] + data
    
    with open(fn, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

def basicTblSgn(caption,label,heading,data,TD,headRpl=None,cp=0,r0=0,l0=1,):
    """Creates a latex-style table.
    
    NB Based on 'basicTable' from 'dss_python_funcs.py'
    creates a simple table. caption, label, TD (table directory) are strings;
    heading is a list of strings, and data is a list of lists of strings,
    each sublist the same length as heading.
    
    See also: basicTblDict
    
    Inputs
    ---
    caption: the table caption
    label: the table label, i.e., \ref{t:label}. use l0=0 to cancel this.
    heading: The headings for each row of the table
    data: a list of lists of strings to populate the table
    TD: the table directory.
    headRpl: use this to replace the heading prior to saving. Useful when
            creating more complex tables with multiple rows etc.
    cp: caption Position - if 1, sets to bottom, otherwise top.
    r0: if True, return the first column aligned right
    l0: if False, then do not put a label in (assume label put in main tex doc).

    Returns
    ---
    latexText - the text written to TD + label + '.tex'.
    
    """
    if not(TD[-1]=='\\'):
        TD = TD+'\\'
    
    if headRpl is None:
        headTxt = ''
        for head in heading:
            headTxt = headTxt + head + ' & '
        
        headTxt = headTxt[:-3]
        headTxt = headTxt + ' \\\\\n'
    else:
        headTxt = headRpl
    
    nL = 'r'+(len(heading) - 1)*'l' if r0 else len(heading)*'l'

    lbl = '}\\label{t:'+label+'}\n' if l0 else '}\n'
    
    dataTxt = ''
    for line in data:
        if len(line)!=len(heading):
            print('\nWarning: length of line does not match heading length.\n')
        for point in line:
            dataTxt = dataTxt + point + ' & '
        dataTxt = dataTxt[:-3]
        dataTxt = dataTxt + ' \\\\\n'
    
    if cp==0:
        latexText = '% Generated using basicTblSgn.\n\\centering\n\\caption{'\
                    +caption+lbl+'\\begin{tabular}{'\
                        +nL+'}\n\\toprule\n'+headTxt+'\\midrule\n'\
                        +dataTxt+'\\bottomrule\n\\end{tabular}\n'
    elif cp==1:
        latexText = '% Generated using basicTblSgn.\n\\centering\n'\
                    '\n\\begin{tabular}{'\
                        +nL+'}\n\\toprule\n'+headTxt+'\\midrule\n'\
            +dataTxt+'\\bottomrule\n\\end{tabular}\n\\caption{'+caption+lbl
    
    with open(TD+label+'.tex','wt') as handle:
        handle.write(latexText)
    return latexText

def saveDataFunc(fn,mode='pkl',**kwargs):
    """A simple function for saving data as a .pkl file in a standard way.
    
    The data is picked straight into fn - this function is mostly used to
    provide a nice standard interface for later.
    
    NB - copied from MiscUtils from the Sprint.
    
    Inputs
    -----
    - mode: 'pkl' (default) or 'csv'. Former is one file, later has a seperate
            readme text file.
    
    kwargs
    --
    Metadata options:
    - readme: string 
    
    Data options:
    - data: numpy array, preferably of FLOAT/INT data
    - dataHead: provide with data, listing the column headings
    # - flags: numpy array of BOOL data # <- out of use
    # - flagsHead: provide with flags to list column headings # <- out of use
    
    Time series time options:
    - tStamps: full timestamps numpy array, in terms of python datetime obj
    - tDict: option to save as {t0,t1,dt} dictionary to save on memory
            (use tDict2stamps func (above) to convert back to tStamps)
    
    Note that this only has very rudimentary error checking at this stage.
    """
    keysAllowed = {
                   'readme',
                   'data',
                   'dataHead',
                   'tStamps',
                   'tDict',
                  }
    assert(set(kwargs.keys()).issubset(keysAllowed))
    
    if fn.split('.')[-1]!=mode:
        fn0 = fn
        fn += '.'+mode
    else:
        fn0 = fn.split('.')[0]
    
    if mode=='pkl':
        with open(fn,'wb') as file:
            pickle.dump(kwargs,file)
            print(f'\nData written to:\n--->\t{fn}\n')
    
    elif mode=='csv':
        head = ['IsoDatetime'] + kwargs.get('dataHead',['data'])
        if 'tDict' in kwargs.keys():
            tt = tDict2stamps(kwargs['tDict'])
        elif 'tStamps' in kwargs.keys():
            tt = kwargs['tStamps']
        
        data = kwargs['data']
        if data.ndim==1:
            data = data.reshape((-1,1))
        
        csvData = [[t.isoformat()]+x.tolist() 
                                        for t,x in zip(tt,data)]
        data2csv(fn,csvData,head)
        
        if 'readme' in kwargs.keys():
            with open(fn0+'_readme.txt','w') as file:
                file.write(kwargs['readme'])
        
        print(f'\nData written to:\n--->\t{fn}\n')

def tDict2stamps(tDict):
    """Parameter tDict has keys t0, t1, dt.
    
    Uses tset2stamps to build the time set."""
    return tset2stamps(tDict['t0'],tDict['t1'],tDict['dt'])

def boxplotqntls(qVals,ax=None,**kwargs):
    """A function for plotting boxplots with given quantiles.
    
    Similar to 'fillplot'
    
    Inputs
    ---
    qVals - 5 x N
    ax - the axis to plot onto
    
    kwargs
    ---
    - xPos as the N-length positions of the qVals
    - width as the width of the plots as a function of xPos (0 to 1)
    - lw as the plot linewidths
    - edgecolors as the color[s] of the box edges & whiskers
    - facecolors as the color[s] of the box face (default None)
    - alpha as the alpha of the box face (default 1)
    
    Returns
    ---
    ax - the axis plotted onto
    
    """
    width = kwargs.get('width',0.6)
    lw = kwargs.get('lw',0.8)
    N = len(qVals[0])
    xPos = kwargs.get('xPos',np.arange(N))
    
    ec = kwargs.get('edgecolors',['k']*N)
    fc = kwargs.get('facecolors',['None']*N)
    lph = kwargs.get('alpha',1)
    
    if len(ec)==1:
        ec = [ec]*N
    if len(fc)==1:
        fc = [fc]*N
    
    if ax is None:
        fig,ax = plt.subplots()
    
    if len(xPos)==1:
        dw0 = 1
        if type(ec) is list:
            ec = ec[0]
        if type(fc) is list:
            fc = fc[0]
    else:
        dw0 = width*(xPos[1]-xPos[0])
    
    dw1 = dw0*0.6
    
    ax.vlines(xPos,qVals[0],qVals[1],linewidth=lw,color=ec)
    ax.vlines(xPos,qVals[-2],qVals[-1],linewidth=lw,color=ec)
    ax.hlines(qVals[0],xPos-dw0/2,xPos+dw0/2,linewidth=lw,color=ec)
    ax.hlines(qVals[-1],xPos-dw0/2,xPos+dw0/2,linewidth=lw,color=ec)
    
    # plot the box and median
    ax.hlines(qVals[2],xPos-dw1/2,xPos+dw1/2,linewidth=lw,color=ec)
    
    boxes = [Rectangle((xPos[i]-(dw1/2),qVals[1][i]),width=dw1,
               height=qVals[3][i]-qVals[1][i],angle=0,linewidth=lw,
               facecolor=fc[i],edgecolor=ec[i],alpha=lph) for i in range(N)]
    
    # Why: see SO "How do I set color to Rectangle in Matplotlib?"
    for box in boxes:
        ax.add_artist(box,)
    
    return ax

def data2csv(fn,data,head=None):
    """Write list of lists 'data' to a csv.
    
    If head is passed, put as the first row.
    
    See also: csvIn.
    """
    if not head is None:
        data = [head] + data
    
    with open(fn, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

class struct(object):
    # from https://stackoverflow.com/questions/1878710/struct-objects-in-python
    pass

class structKeyword(struct):
    def __init__(self,kw=[],kwDict={}):
        self.kw = kw
        self.kwInit(kwDict)
    def kwInit(self,kwDict):
        for key,val in kwDict.items():
            if key in self.kw:
                setattr(self,key,val)
            else:
                print('Not setting key-value pair to Lin Model:',key,val)
        for key in self.kw:
            if not hasattr(self,key):
                setattr(self,key,None)
    
    def additem(self,kw,kwval):
        self.kw = self.kw+[kw]
        setattr(self,kw,kwval)
    
    def setitem(self,kw,kwval):
        if kw in self.kw:
            setattr(self,kw,kwval)
        else:
            raise AttributeError('No keyword of type', kw)
    
    def __getitem__(self,keys):
        """k[key] <=> k.key
        
        Is 'overloaded' to allow a set of keys, as in k[a,b] <=> k.a.b,
        or k[(a,b,)] = k.a.b
        """
        if type(keys) is tuple:
            val = getattr(self,keys[0])
            for k in keys[1:]:
                val = getattr(val,k)
        else:
            val = getattr(self,keys)
        return val
    
    def __setitem__(self,key,val):
        """Defined to allow self[key] = val"""
        if type(key) is tuple:
            obj = self[key[:-1]]
            setattr(obj,key[-1],val)
        else:
            setattr(self,key,val)
    
    def asdict(self):
        return {key:getattr(self,key) for key in self.kw}
    
    def __iter__(self):
        return iter(self.asdict().items())

class structDict(structKeyword):
    """Enter a kwDict [kw:val] to create with __init__.
    
    """
    def __init__(self,kwDict):
        self.kw = list(kwDict.keys())
        self.kwInit(kwDict)
    
    def setitem(self,kw,kwval):
        super().setitem(kw,kwval)
    
    def sel_tt_v(self,t0,t1,vv,mode='ex',):
        """Return a structDict with self.x, self.t between [t0,t1] args.
        
        The value is then in self.x rather than self.vv for convenience.
        
        Mode as 'ex' (t1 exclusive) or 'in' (t1 inclusive)
        """
        i0,i1_ = [np.argmax(self.t==_t) for _t in [t0,t1]]
        if mode=='ex':
            i1 = i1_
        elif mode=='in':
            i1 = i1_+1
        
        return structDict({'x':self[vv][i0:i1],'t':self.t[i0:i1]})

def whocalledme(depth=2):
    """Use traceback to find the name (string) of a function that calls it.
    
    Depth:
    1 - this function
    2 - is the default number, finds the function that calls this function
    3 - the function above that
    4 - the function above that (etc...)
    """
    names = []
    for line in traceback.format_stack():
        whereNwhat = line.strip().split('\n')[0]
        names.append(whereNwhat.split(' ')[-1])
    return names[-depth]

def nanlt(a,b):
    """Find a < b elementwise, for a and b with nan elements.
    
    Dual is nangt(a,b)
    """
    mask = np.zeros(a.shape,dtype=bool)
    np.less(a,b,out=mask,where=~(np.isnan(a)+np.isnan(b)))
    return mask

def nangt(a,b):
    """Find a > b elementwise, for a and b with nan elements.
    
    Dual is nanlt(a,b)
    """
    mask = np.zeros(a.shape,dtype=bool)
    # np.greater(a,b,out=mask,where=~np.isnan(a))
    np.greater(a,b,out=mask,where=~(np.isnan(a)+np.isnan(b)))
    return mask

def plotEdf(x,*p_args,xlms=None,ax=None,mm='norm',**p_kwargs):
    """Plot the empirical distribution function.
    
    INPUTS
    ------
    x: the thing to plot the EDF of,
    p_args: normal plot arguments to go into 'step' as *p_args
    p_kwargs: plot keyword arguments to go into 'step' as **p_kwargs
    xlms: x limits to go from/to
    ax: the axis to plot onto
    mm: 'norm' for a normal EDF or '%' for percentage
    
    RETURNS
    -------
    ax: the axis this was plotted onto,
    [xlo,xhi]: the low/high x limits that the step plot goes to.
    
    """
    if ax is None:
        fig, ax = plt.subplots()
    x = np.sort(x)
    y = np.arange(0,len(x)+1)/len(x)
    if xlms is None:
        xhi = x[-1] + (x[-1] - x[0])*0.2
        xlo = x[0] - (x[-1] - x[0])*0.2
    else:
        xlo, xhi = xlms
    xs = np.r_[xlo,x,xhi]
    ys = np.r_[y,1]
    if mm=='%':
        ys = ys*100
    
    plt.step(xs,ys,where='post',*p_args,**p_kwargs)
    plt.xlim((xlo,xhi))
    return ax, [xlo,xhi]

def cdf2qntls(xcdf,fcdf,pvals=np.linspace(0,1,5)):
    """Calculate the values of x at the pvals using a CDF function xcdf.
    
    Inputs
    ---
    xcdf as argument values of the cdf
    fcdf as the corresponding cdf function values
    pvals as the probability values to be calculated at
    
    Returns
    ---
    xVals, the vales of x at the pvals
    
    """
    pvals = [pvals] if type(pvals) is float else pvals
    idxs = [np.argmax(fcdf>pval) for pval in pvals]
    return [xcdf[idx] for idx in idxs]

def rerr(A,A_,ord=None,p=True):
    """Calculate rerr = norm(A-A_)/norm(A). p input for 'print' value as well.
    
    NB: the order of A, A_ DOES matter! 'A' is the 'Truth'.
    
    A, A_ should have the same type (sparse or not), no error checking on this.
    
    Inputs
    A: the true A
    A_: the approx A
    ord: passed into norm (ignored if one dimensional)
    p: bool, 'print' flag
    
    Returns
    ---
    rerr - as above.
    """
    if sparse.issparse(A):
        rerr = sparse.linalg.norm(A-A_,ord=ord)/sparse.linalg.norm(A,ord=ord)
    elif type(A) in [float, int, np.float64]:
        rerr = np.abs(A - A_)/np.abs(A)
    else:
        rerr = norm(A-A_,ord=ord)/norm(A,ord=ord)
    
    if p: 
        print(rerr)
    return rerr

class sctXt(struct):
    # A very basic structure holding t, x values (ie primitive time series)
    def __init__(self,x=None,t=None):
        self.x = x
        self.t = t
    
    def sel_tt(self,t0,t1,mode='ex',):
        """Return a sctXt object with same data but from t0 to t1 (excl.)."""
        i0,i1_ = [np.argmax(self.t==_t) for _t in [t0,t1]]
        if mode=='ex':
            i1 = i1_
        elif mode=='in':
            i1 = i1_+1
        
        return sctXt(self.x[i0:i1],self.t[i0:i1],)

def mtOdict(ll):
    """Create an odict with not-linked lists as elements for each key in ll."""
    od = odict()
    [od.__setitem__(k,[]) for k in ll]
    return od

def aMulBsp(a,b):
    """Returns a.dot(b) if b is sparse, as a numpy array."""
    val = (b.T.dot(a.T)).T
    if sparse.issparse(val):
        val = val.toarray()
    return val