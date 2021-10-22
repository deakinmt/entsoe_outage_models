""" Datasets from entsoe-py-data.py. 

The data is copy-pasted from the doc from
https://github.com/EnergieID/entsoe-py
Last updated: 20-3-20

The up-to-date big table can be gotten from
https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html#_reference_documentation

----------- License FROM https://github.com/EnergieID/entsoe-py

MIT License

Copyright (c) 2017 EnergieID cvba-so

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
------------

"""
# From API guide, A.10 (above)
# 10Y1001A1001A59C Ireland (SEM) BZ / MBA
# 10YIE-1001A00010 Ireland, EirGrid CA

DOMAIN_MAPPINGS = {
    'AL': '10YAL-KESH-----5',
    'AT': '10YAT-APG------L',
    'BA': '10YBA-JPCC-----D',
    'BE': '10YBE----------2',
    'BG': '10YCA-BULGARIA-R',
    'BY': '10Y1001A1001A51S',
    'CH': '10YCH-SWISSGRIDZ',
    'CZ': '10YCZ-CEPS-----N',
    'DE': '10Y1001A1001A83F',
    'DK': '10Y1001A1001A65H',
    'EE': '10Y1001A1001A39I',
    'ES': '10YES-REE------0',
    'FI': '10YFI-1--------U',
    'FR': '10YFR-RTE------C',
    'GB': '10YGB----------A',
    'GB-NIR': '10Y1001A1001A016',
    'GR': '10YGR-HTSO-----Y',
    'HR': '10YHR-HEP------M',
    'HU': '10YHU-MAVIR----U',
    'I0': '10YIE-1001A00010', # <<< the 'old' Irish data.
    'IE': '10Y1001A1001A59C',
    'IT': '10YIT-GRTN-----B',
    'LT': '10YLT-1001A0008Q',
    'LU': '10YLU-CEGEDEL-NQ',
    'LV': '10YLV-1001A00074',
    # 'MD': 'MD',
    'ME': '10YCS-CG-TSO---S',
    'MK': '10YMK-MEPSO----8',
    'MT': '10Y1001A1001A93C',
    'NL': '10YNL----------L',
    'NO': '10YNO-0--------C',
    'PL': '10YPL-AREA-----S',
    'PT': '10YPT-REN------W',
    'RO': '10YRO-TEL------P',
    'RS': '10YCS-SERBIATSOV',
    'RU': '10Y1001A1001A49F',
    'RU-KGD': '10Y1001A1001A50U',
    'SE': '10YSE-1--------K',
    'SI': '10YSI-ELES-----O',
    'SK': '10YSK-SEPS-----K',
    'TR': '10YTR-TEIAS----W',
    'UA': '10YUA-WEPS-----0',
    'DE-AT-LU': '10Y1001A1001A63L', # seems to be used pre oct 2018
    'DE-LU': '10Y1001A1001A82H', # seems to be used post oct 2018
}

BIDDING_ZONES = DOMAIN_MAPPINGS.copy()
BIDDING_ZONES.update({
    'DE': '10Y1001A1001A63L',  # DE-AT-LU
    'LU': '10Y1001A1001A63L',  # DE-AT-LU
    'DE18': '10Y1001A1001A82H',  # DE-LU
    'IT-NORD': '10Y1001A1001A73I',
    'IT-CNOR': '10Y1001A1001A70O',
    'IT-CSUD': '10Y1001A1001A71M',
    'IT-SUD': '10Y1001A1001A788',
    'IT-FOGN': '10Y1001A1001A72K',
    'IT-ROSN': '10Y1001A1001A77A',
    'IT-BRNN': '10Y1001A1001A699',
    'IT-PRGP': '10Y1001A1001A76C',
    'IT-SARD': '10Y1001A1001A74G',
    'IT-SICI': '10Y1001A1001A75E',
    'NO-1': '10YNO-1--------2', 
    'NO-2': '10YNO-2--------T',
    'NO-3': '10YNO-3--------J',
    'NO-4': '10YNO-4--------9',
    'NO-5': '10Y1001A1001A48H',
    'SE-1': '10Y1001A1001A44P',
    'SE-2': '10Y1001A1001A45N',
    'SE-3': '10Y1001A1001A46L',
    'SE-4': '10Y1001A1001A47J',
    'DK-1': '10YDK-1--------W',
    'DK-2': '10YDK-2--------M'
})


PSRTYPE_MAPPINGS = {
    'A03': 'Mixed',
    'A04': 'Generation',
    'A05': 'Load',
    'B01': 'Biomass',
    'B02': 'Fossil Brown coal/Lignite',
    'B03': 'Fossil Coal-derived gas',
    'B04': 'Fossil Gas',
    'B05': 'Fossil Hard coal',
    'B06': 'Fossil Oil',
    'B07': 'Fossil Oil shale',
    'B08': 'Fossil Peat',
    'B09': 'Geothermal',
    'B10': 'Hydro Pumped Storage',
    'B11': 'Hydro Run-of-river and poundage',
    'B12': 'Hydro Water Reservoir',
    'B13': 'Marine',
    'B14': 'Nuclear',
    'B15': 'Other renewable',
    'B16': 'Solar',
    'B17': 'Waste',
    'B18': 'Wind Offshore',
    'B19': 'Wind Onshore',
    'B20': 'Other',
    'B21': 'AC Link',
    'B22': 'DC Link',
    'B23': 'Substation',
    'B24': 'Transformer',
}


# PSR types which (for purposes here) are 'intermittent'
PSRTYPE_MAPPINGS_INTER = {
    'B11': 'Hydro Run-of-river and poundage',
    'B16': 'Solar',
    'B18': 'Wind Offshore',
    'B19': 'Wind Onshore',
}


DOCSTATUS = {
    'A05': 'Active',
    'A09': 'Cancelled',
    'A13': 'Withdrawn'
}

BSNTYPE = {'A29': 'Already allocated capacity (AAC)',
           'A43': 'Requested capacity (without price)',
           'A46': 'System Operator redispatching',
           'A53': 'Planned maintenance',
           'A54': 'Unplanned outage',
           'A85': 'Internal redispatch',
           'A95': 'Frequency containment reserve',
           'A96': 'Automatic frequency restoration reserve',
           'A97': 'Manual frequency restoration reserve',
           'A98': 'Replacement reserve',
           'B01': 'Interconnector network evolution',
           'B02': 'Interconnector network dismantling',
           'B03': 'Counter trade',
           'B04': 'Congestion costs',
           'B05': 'Capacity allocated (including price)',
           'B07': 'Auction revenue',
           'B08': 'Total nominated capacity',
           'B09': 'Net position',
           'B10': 'Congestion income',
           'B11': 'Production unit'}

DOCUMENTTYPE = {'A09': 'Finalised schedule',
                'A11': 'Aggregated energy data report',
                'A25': 'Allocation result document',
                'A26': 'Capacity document',
                'A31': 'Agreed capacity',
                'A44': 'Price Document',
                'A61': 'Estimated Net Transfer Capacity',
                'A63': 'Redispatch notice',
                'A65': 'System total load',
                'A68': 'Installed generation per type',
                'A69': 'Wind and solar forecast',
                'A70': 'Load forecast margin',
                'A71': 'Generation forecast',
                'A72': 'Reservoir filling information',
                'A73': 'Actual generation',
                'A74': 'Wind and solar generation',
                'A75': 'Actual generation per type',
                'A76': 'Load unavailability',
                'A77': 'Production unavailability',
                'A78': 'Transmission unavailability',
                'A79': 'Offshore grid infrastructure unavailability',
                'A80': 'Generation unavailability',
                'A81': 'Contracted reserves',
                'A82': 'Accepted offers',
                'A83': 'Activated balancing quantities',
                'A84': 'Activated balancing prices',
                'A85': 'Imbalance prices',
                'A86': 'Imbalance volume',
                'A87': 'Financial situation',
                'A88': 'Cross border balancing',
                'A89': 'Contracted reserve prices',
                'A90': 'Interconnection network expansion',
                'A91': 'Counter trade notice',
                'A92': 'Congestion costs',
                'A93': 'DC link capacity',
                'A94': 'Non EU allocations',
                'A95': 'Configuration document',
                'B11': 'Flow-based allocations'}