import os, sys
from datetime import datetime, timedelta

# Start and end download dates
dstart = datetime(2017,10,1)
dend = datetime(2020,4,1) # exclusive
dstart_xtra = datetime(2016,10,1)
dend_xtra = datetime(2021,4,1) # inclusive
dT = timedelta(0,24*3600)

token = r'7b53f33d-c817-4f1c-9ed5-599b5501c529'