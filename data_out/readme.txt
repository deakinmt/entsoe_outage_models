Outage data for NW European countries, Oct 2016-Mar 2021.

Data downloaded from ENTSOe's transparency platform using the 
entsoe_outage_models repository, __LINK_HERE__ [1], supported by
the Supergen Energy Networks Hub's CLEARHEADS Flex Fund project.


M. Deakin, November 2021
Contact: matthew.deakin@newcastle.ac.uk

Fields
---
1. Time - datetime, UTC
2. Planned outages, MW
3. Forced outages, MW
4. Planned outage 'uncertainty', MW
5. Forced outage 'uncertainty', MW

Planned and forced outages (2/3) are defined as in [2], article 15. 

The country-wide values of each of these are determined by
aggregating the outages across zones (e.g., for Norway,
we use NO-1 - NO-5).

The 'uncertainty' in 4/5 refers to the fact that some individual
outage reports provide unclear information as to the state
of a generator at a given time. Rather than providing a set of
rules to reconcile these, the maximum and minimum possible outage
was determined for each zone; the average of these was taken for 
2/3; the uncertainty is therefore +/- 4, 5 for 2, 3 respectively.

Note, however, that small outages (less than 100 MW) are 
required [2]. It can be seen in the outage reports on the
transparency platform that some of these are reported, but
others are not. So, 'uncertainty' should be taken into consideration
in the strict sense above, *NOT* in terms of the true uncertainty
in the outage rates of the countries.


References
---
[1] __LINK_TO_GITHUB_HERE__
[2] Commission Regulation (EU) No 543/2013 of 14 June 2013 on 
submission and publication of data in electricity markets and 
amending Annex I to Regulation (EC) No 714/2009 of the European 
Parliament and of the Council Text with EEA relevance. 
https://transparency.entsoe.eu/content/static_content/Static%20content/knowledge%20base/data-views/outage-domain/Data-view%20Unavailability%20of%20Production%20and%20Generation%20Units.html
