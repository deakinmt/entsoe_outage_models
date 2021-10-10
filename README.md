# entsoe_outage_models

A package for building time-series models of generator outages
based on NW European countries.



Scripts / options
---
download_config.py - various information for changing download options,
    including the token used to call the API.
entsoeProductionDownload.py - A script to downnload production and 
    generation units


Directories
---
apiDoc - various bits of documentation from entsoe's API website
entsoeData - firectory which entsoe data is downloaded into
misc - miscellaneous


Other
---
import_list.py - list of imports for the various scripts above
eom_funcs.py - functions for working with the above scripts
eom_utils.py - various util functions
entsoe_py_data.py - various useful codes for use with the API,
    based on the useful API client entsoe-py


Acronyms
---
- eom = entsoe_outage_models
- png = Production aNd Generation
- CH = ClearHeads (project)
- BZ = Balancing Zone
