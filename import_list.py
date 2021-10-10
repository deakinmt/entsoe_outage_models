import os, sys, csv, pickle, types, zipfile
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
from pprint import pprint
from matplotlib import cm
from collections import OrderedDict as odict
from bunch import Bunch
from copy import copy, deepcopy
import matplotlib.dates as mdates
from progress.bar import Bar

from eom_utils import repl_list, fn_root, data2csv, structDict, sctXt,\
        tDict2stamps, boxplotqntls, nanlt, nangt, plotEdf, cdf2qntls, rerr
exec(repl_list)

import eom_funcs as eomf
reload(eomf)
getCodes = eomf.getCodes
getUrl = eomf.getUrl
reqData = eomf.reqData
processXml = eomf.processXml
reqInfo = eomf.reqInfo
ETnse = eomf.ETnse

wd = sys.path[0]
sd = os.path.join(wd,'entsoeData')

twh2mcm = 1e3/10.55
from entsoe_py_data import PSRTYPE_MAPPINGS
reqTypes = {
            0:'dap',
            1:'actlTotLd',
            2:'fcstTotLd',
            3:'wNsF',
            4:'B0630_BTA60', # Doesn't seem to be working?
            5:'B0630_BTA61', # Doesn't seem to be working?
            6:'wNs',
            7:'unvblGp', # GP: Generation, planned. [see ./entsoeOutage.py]
            8:'icppu', # installed capacity per production unit
            9:'png', # production and generation units
            }

plt.style.use(os.path.join(fn_root,'misc','styles','ieee.mplstyle'))
import xml.etree.ElementTree as ET