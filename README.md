# entsoe_outage_models

A package for building empirical and analytic time-series models of
the aggregated unavailability of power generation fleets.

Main usage of this code has been in the creation of time series of 
aggregated generator outages for a number of Northwest European 
countries using the ENTSOe Transparency Platform, although code
for generating analytic models of fleet unavailabilities is also 
provided.

The aggregated generator unavailability data for NW European 
countries is available at:
https://doi.org/10.25405/data.ncl.18393971

This code was written for the analysis undertaken in the paper
"Comparing Generator Unavailability Models with Empirical 
Distributions from Open Energy Datasets" submitted to the PMAPS
2022 conference (arxiv link to follow).

Note
---
In general, the scripts use options to select what to run - e.g.,
entsoeOutageDownload.py line  17-24 can be changed from 0 to 1
to enable those options.

Workflow
---
It is suggested that the usage is approached in the following way.
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

Limitations with the ENTSOe outage data
---
L1. By default, 'outages' which result in the output being 1.33
  times greater than the generator nominal power are neglected.
L2. Even with L1, sometimes 'negative' outages occur (e.g., for
  DK in early 2020, I0 in early 2019).
L3. By default, we ignore outages on wind/solar/hydro run of river.
L4. The reasons/convention for individual generators and what is
  classed as a 'Forced' versus 'Planned' outage has not been
  considered.
L5. Forced and planned outages are considered seperately; possible
  clashes between these are not considered (e.g., both forced and
  planned outages occuring at the same time).
  
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

Directories
---
apiDoc - various bits of documentation from entsoe's API website
entsoeData - directory which entsoe data is downloaded into
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


Acknowledgements
---
This work was funded by the Supergen Energy Network hub (EP/S00078X/2)
through the Climate-Energy Modelling for Assessing Resilience: Heat 
Decarbonisation and Northwest European Supergrid (CLEARHEADS) project.

