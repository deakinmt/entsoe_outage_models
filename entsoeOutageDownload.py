import os, sys
with open(os.path.join(sys.path[0],'import_list.py')) as file:
    exec(file.read())


from eom_funcs import getOutageData

# Config parameters
from download_config import dstart, dend, dstart_xtra, dend_xtra, dT

codes = getCodes('CH') # change this to change how much data is downloaded
codes = getCodes(['GB', 'IE', 'BE', 'NL', 'I0', 'FR','DK-1', 'DK-2', 
                            'NO-1', 'NO-2', 'NO-3', 'NO-4', 'NO-5',])

sdOut_ = os.path.join(sd,'outage_data')

i0 = 0 # tweak if bailed previously due to, e.g., failed connection
dset = np.arange(dstart_xtra,dend_xtra,dT,dtype=object)

# Useful text strings
txt0 = "No matching data found for Data item Planned Unavailability of Generation Units [15.1.A], Changes in Actual Availability of Generation Units [15.1.B]"
txt_rtn = 'The amount of requested data exceeds allowed limit'

# Call the API
for ctry,code in codes.items():
    # Make directory if there isn't one
    sd_ = os.path.join(sdOut_,ctry[:2])
    _ = os.mkdir(sd_) if not os.path.exists(sd_) else None
    
    # Then look through each day
    for d in dset[i0:]:
        print(f'Getting date: {d}')
        # Try with first day
        urls,rs,scs = eomf.getOutageData(code,d,dT,)
        
        # If 400 returned, go with the offsets
        if rs[0].status_code==400:
            mssg = rs[0].content.decode('utf-8')
            if txt_rtn in mssg:
                # Assume number between 200 and 10000
                doc_no = int(mssg.split('requested: ')[1][:4])
                n_batches = 1 + doc_no//200
                urls,rs,scs = eomf.getOutageData(code,d,dT,n_batches,)
        
        # Find out if successful
        success = all([sc==200 for sc in scs])
        
        # If not successful due to no data, continue
        if txt0 in rs[0].text and not success:
            print('No data - continuing with no save.')
            continue
        
        # If we get a different error code, cancel
        bail = any([sc not in [200,400] for sc in scs])
        if bail:
            print(f'Bailed at d: {d} ; i0 = {(d - dset[0]).days}')
            raise Exception(f'-- Returned: {scs}')
        
        # Otherwise continue
        for ii,r in enumerate(rs):
            nm = f'unvblGp_{ctry}_{eomf.d2s(d)}_{ii+1}_{len(rs)}.zip'
            with open(os.path.join(sd_,nm),'wb') as file:
                file.write(r.content)
