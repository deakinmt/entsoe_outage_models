import os, sys
from datetime import datetime, timedelta

# Start and end download dates
dstart = datetime(2017,10,1)
dend = datetime(2020,4,1) # exclusive
dstart_xtra = datetime(2016,10,1)
dend_xtra = datetime(2021,4,1) # inclusive
dT = timedelta(0,24*3600)

token = __your_entsoe_token_here__ 
# e.g., token = r'1a23a45a-a678-9a1a-2aa3-456a7891a234'