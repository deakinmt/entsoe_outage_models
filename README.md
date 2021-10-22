# entsoe_outage_models

A package for building time-series models of generator outages
based on NW European countries.

===
TO DO
---
[x] how to most suitably load in the require production data
[ ] add an option if the installed_prod has not been saved
[x] check we have all countries 
[x] check if vvmult should be used
[x] checksums on all country outputs
[ ] checksums on intermediate outputs (e.g., vvmult)
[ ] exploring the flat period GB Xmas 2017
[x] exploring the errors in GB data
[x] exploring the negative values in the IE data
[x] exploring discontinuity in DE, NL data
[ ] saving the individual generator outages sparse matrix
[x] saving the output data for re-use
[x] exploring the min-max errors
[ ] looking at outages of only conventional plant, not renewables
===


Notes
---

not clear at the moment if the data can be clearly reused:
https://transparency.entsoe.eu/content/static_content/download?path=/Static%20content/terms%20and%20conditions/191025_List_of_Data_available_for_reuse_v2_cln.pdf

In general, the scripts use options to select what to run - e.g.,
entsoeOutageDownload.py line  17-24 can be changed from 0 to 1
to enable those options.

I0 refers to __, whilst IE refers to the 'old' irish datacode,
whilst IE refers to the full SEM.


Scripts / options
---
entsoeOutageDownload.py - a script for automating the download of 
    outage data for individual bidding zones
entsoeOutage.py - a script for processing the outage data into .pkl
    files for onward analysis (e.g., creating a time series of
    data), as well as exploring the production & generation units.
download_config.py - various information for changing download options,
    including the token used to call the API.
entsoeProductionDownload.py - A script to downnload production and 
    generation units


Workflow
---
It is suggested that the data is approached in the following way:
1. The outage data for individual countries is download from 
  entsoeOutageDownload.py, using download_config.py.
2. The production and generation units are downloaded from
  entsoeProductionDownload.py.
3. The data is then explored using the options in entsoeOutage.py.
4. If wanting to use 'true' generator sizes, then there is a manual
  step. First, log into the online platform. Then, go to the "Installed
  Capacity per production Type" page; select years 2015-2021 for each
  of [BE, DE-AT-LU, DE-LU, DK-1, DK-2, FR, GB, IE, NL, NO-1, NO-2, NO-3
  NO-4, NO-5], select export as csv. These should be downloaded and 
  saved with the balancing zone code prepended the file, e.g.,
  "BE_Installed Capacity per Production Type_201501010000-202201010000";
  These should then be saved in the directory .\entsoe_outage_data\misc\inpr\.


Directories
---
apiDoc - various bits of documentation from entsoe's API website
entsoeData - firectory which entsoe data is downloaded into
gallery - directory to save figures by default
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
- APs = Availability Periods
- DPs = Datetime [Availability] Periods (reconciled APs for a single BZ/country)
- ECR = [NGESO EMR Delivery Body's] Electricity Capacity Report

