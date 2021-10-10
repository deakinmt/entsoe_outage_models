"""entsoeProductionDownload.py

A script for downloading the production units for each region that is a part
of the Clearheads NW Europe zone, calculated monthly.

Note that this data does not appear to be the most accurate source of data
as to the 'real' running generators in a given region; it is presented as 
indicative data only.

"""
import os, sys, pickle, types, zipfile, time
from collections import OrderedDict as odict

with open(os.path.join(sys.path[0],'import_list.py')) as file:
    exec(file.read())

# Flag to save an XML file of the downloaded data for inspection
xmlDump = 0

# Get the market areas to use
codes = getCodes('CH') # change this to change how much data is downloaded
_ = [codes.pop(k) for k in ['',]] # list countries/zones not wanted here.

# Choose the months wanted downloading
dset = [datetime(yr,mo,1) for yr in range(2015,2021) for mo in range(1,13)]

# Function used to create API call
def getProductionUrl(code,d0):
    """Get the url for outage data from d0 to d1."""
    url = getUrl('png',code,2018,opts=[[None]])
    url = url.replace('__datehere__',eomf.m2s(d0),)
    return url

# Call the API
for ctry,code in codes.items():
    # Make directory if there isn't one
    sd_ = os.path.join(sd,'png',ctry[:2])
    _ = os.mkdir(sd_) if not os.path.exists(sd_) else None
    
    # Then look through each month
    for d in dset:
        print(f'Getting {ctry} date: {d}')
        
        url = getProductionUrl(code,d)
        r = reqData(url)
        if r.status_code!=200:
            raise Exception('200 not returned!')
        
        head, data = eomf.process_production(r)
        
        fnOut = os.path.join(sd_,f'png_{ctry}_{eomf.m2s(d)}.csv')
        data2csv(fnOut,data,head=head)

if xmlDump:
    with open(os.path.join(sd,'b_dump.xml'),'w',encoding='utf-8') as file:
        file.write(r.content.decode('utf-8'))
