import os, sys, pickle, types, zipfile
import pandas as pd
from statsmodels.tsa.stattools import acf

from timeit import default_timer as timer

with open(os.path.join(sys.path[0],'import_list.py')) as file:
    exec(file.read())

# Config parameters
from download_config import dstart, dend, dstart_xtra, dend_xtra, dT
ds, de = dstart_xtra, dend_xtra
from scipy import sparse
from eom_utils import aMulBsp
fig_sd = r"C:\Users\nmd155\OneDrive - Newcastle University\papers\pmaps2022\outage_paper\figures"
tbl_sd = r"C:\Users\nmd155\OneDrive - Newcastle University\papers\pmaps2022\outage_paper\tables"

fs_sngl_ = (4.8,2.8)
fs_dbl = (2.8,2.8)

# Select the countries to use
ccs = ['GB', 'IE', 'I0', 'BE', 'NL', 'FR', 'DK', 'ES', 'NO', 'DE',]

# Process the data to create the 
process_APs = 0
process_DPs = 0
save_outputs = 0

# Plotting options
fig_entsoeout = 0 # change cc below to change output
fig_entsoePs = 0
fig_entsoePs_minmax = 0
fig_entsoePs_ratio = 0
fig_pngUnits = 0
fig_hydro = 0

# Misc figures
pltTsGenerator = 0
pltTsCcWinters = 0
pltOutageChanges = 0
pltOutageChanges_v2 = 0
tbl_fuel_mttrs = 0
pltTsXmplRuns = 0
pltOneWeekXmpl = 0
tbl_acf = 0

# Plotting examples of how the algorithm works
plt_approach_xmpl_vvmults = 0
plt_approach_xmpl_contiguous = 0
plt_xmpl_1 = 0
plt_xmpl_2 = 0
plt_xmpl_3 = 0
plt_xmpl_4 = 0
plt_xmpl_5 = 0
plt_xmpl_n = 0

# Save figure flag
sf = 0

# Script options
rerun = 0
cc = 'GB' # opts - 'GB', 'IE', 'I0', 'BE', 'NL', 'FR', 'ES', 'DK', 'DE',
av_model = 'ecr'

# Load the data and get the clock & keys
APs, mm, kks, kksD = eomf.load_aps(cc,sd,rerun=rerun,save=True)
psrn2id = {mm[k]['psrName']:k for k in mm.keys()}

# Note: using the outage generator requires data to be downloaded
# as described in the readme.
self = eomf.bzOutageGenerator(av_model=av_model,)

if pltOutageChanges:
    drange,dpsX,dpsXx,dpsXr = eomf.load_dps(ds,de,'GB',sd,rerun=False)
    for xx,nm in zip([dpsXr['f'],dpsXr['t']],['Forced','Total',]):
        dpss = [self.getWinters(sctXt(t=drange,x=xx),yrStt=yr,nYr=1,) 
                                                    for yr in range(2016,2021)]
        
        qntl = 1 - (1/24)
        hist_bins = np.arange(-4050,4050,100)
        for ii in [1,2,4,6,12,18,24,]:
            dF = np.concatenate([dps.x[ii:] - dps.x[:-ii] for dps in dpss])
            qq = np.nanquantile(dF,qntl)
            plt.hist(dF,bins=hist_bins,histtype='step',
                                        label=f'{ii} hr ({qq:.0f} MW)', lw=0.9)
        
        plt.xlabel(
            'Power difference, MW (+ve: future \nhour greater outage than now)')
        plt.ylabel('Count',)
        plt.title(f'Changes in {nm} Outages, GB Winters 16/17-20/21')
        plt.legend(title=f'Hour difference\n({qntl:.2%} quantile)',
                                                        fontsize='small',)
        if sf:
            sff(f'pltOutageChanges_{nm}')
        
        tlps()

if pltOutageChanges_v2:
    # Only do GB Forced for now.
    ccsel = 'GB'
    dr1,dpsX,dpsXx,dpsXr = eomf.load_dps(ds,de,ccsel,sd,rerun=False)
    dr2 = np.arange(datetime(2000,11,1),datetime(2021,4,1),timedelta(0,3600,))
    ua = self.build_unavl_model(assign=False,seed=0,nt=len(dr2),)
    mdl = ua[ccsel]['v'].dot(ua[ccsel]['ua'])
    
    hrs = np.arange(24)
    for xx,dr,nm in zip([dpsXr['t'],mdl,],[dr1,dr2],['tot','mdl',]):
        dpss = [self.getWinters(sctXt(t=dr,x=xx),yrStt=yr,nYr=1,) 
                                            for yr in range(2000,2021)]
        
        qntl = 1 - (1/24)
        qntls_ = [1/24,(1/(24*7)),1/(24*30),1/(24*7*20)][::-1]
        qntls = [[qq,1-qq] for qq in qntls_]
        lbls = ['1 per day','1 per week','1 per month','1 per winter',][::-1]
        qqs = [[[0,0]] for i in range (len(qntls))]
        for hr in hrs[1:]:
            dF = np.concatenate([dps.x[hr:] - dps.x[:-hr] for dps in dpss])/1e3
            _ = [qq.append(np.nanquantile(dF,qntl)) 
                                            for qq, qntl in zip(qqs,qntls)]
        
        fig, ax = plt.subplots(figsize=fs_dbl,)
        for ii,(qq,lbl) in enumerate(zip(qqs,lbls)):
            plt.fill_between(hrs,listT(qq)[0],listT(qq)[1],
                            color=cm.Blues((ii+1)/(len(lbls))),label=lbl,)
            _ = [plt.plot(hrs,listT(qq)[i],'k-',lw=0.25) for i in range(2)]
        
        if nm=='tot':
            plt.legend(fontsize='small',)
            
        set_day_label()
        plt.xlim((0,23))
        plt.ylim((-13.5,13.5))
        plt.xlabel('Hours since $\\tau$')
        plt.ylabel(f'Change in {ccsel} Outages (+ve as\n greater outage), GW',)
        if sf:
            sff(f'pltOutageChanges_v2_{nm}',sd=fig_sd,pdf=True,)
        
        tlps()


if process_APs:
    for cc in ccs:
        APs, mm, kks, kksD = eomf.load_aps(cc,sd,rerun=True,save=True)

if process_DPs:
    for cc in ccs:
        drange,dpsX,dpsXx,dpsXr = eomf.load_dps(ds,de,cc,sd,rerun=1,save=1)

save_readme = """Outage data for NW European countries, Oct 2016-Mar 2021.

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
4. Total outages, MW
5. Planned outage 'uncertainty', MW
6. Forced outage 'uncertainty', MW
7. Total outage 'uncertainty', MW

Planned and forced outages (2/3) for an inividual plant are
as defined as in [2], article 15; these fields are the sum of
all generators with either planned or forced outage state.

Total outages (4) is all outages irrespective of outage type.

The country-wide values of each of these are determined by
aggregating the outages across zones (e.g., for Norway,
we use NO-1 - NO-5).

The 'uncertainty' in 5-7 refers to the fact that some individual
outage reports provide ambiguous information as to the state
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
"""

if save_outputs:
    dn_out = os.path.join(fn_root,'data_out',)
    with open(os.path.join(dn_out,'readme.txt'),'w') as file:
        file.write(save_readme)
    
    for cc in ccs:
        drange,dpsX,dpsXx,dpsXr = eomf.load_dps(ds,de,cc,sd,rerun=0,save=0)
        fn = os.path.join(dn_out,f'outages_{cc}',)
        
        data = np.concatenate([
            [dpsXr['p'],],
            [dpsXr['f'],],
            [dpsXr['t'],],
            [dpsXx['p']*0.5,],
            [dpsXx['f']*0.5,],
            [dpsXx['t']*0.5,],
        ]).T
        dataHead = [
            'planned_outages_mw',
            'forced_outages_mw',
            'total_outages_mw',
            'planned_outages_uncertainty_mw',
            'forced_outages_uncertainty_mw',
            'total_outages_uncertainty_mw',
        ]
        saveDataFunc(fn,mode='csv',data=data,dataHead=dataHead,tStamps=drange,)

if fig_entsoePs or fig_entsoePs_minmax or fig_entsoePs_ratio:
    for cc in ccs:
        drange,dpsX,dpsXx,dpsXr = eomf.load_dps(ds,de,cc,sd,rerun=False)
        dps = dpsX['t']
        dps0 = dpsXr['t']
        dps_ = dpsX['t'] + dpsXx['t']
        
        fig, ax = plt.subplots(figsize=fs_sngl_)
        if fig_entsoePs or fig_entsoePs_minmax:
            ylms = (-500,max(dps)*1.15)
            if fig_entsoePs:
                figname = f'fig_entsoePs_{cc}'
                plt.plot_date(drange,dpsXr['p'],'-',label='Planned',lw=0.7,)
                plt.plot_date(drange,dpsXr['f'],'-',label='Forced',lw=0.7,)
                plt.plot_date(drange,dps,'k-',label='Total',lw=0.7,)
            elif fig_entsoePs_minmax:
                figname = f'fig_entsoePs_minmax_{cc}'
                plt.plot_date(drange,dps,'C0-',lw=0.7,label='Min total',)
                plt.plot_date(drange,dps_,'C1-',lw=0.7,label='Max total',)
            
            plt.legend(title=f'Country: {cc}',fontsize='small',)
            plt.ylabel('Outages, MW')
        elif fig_entsoePs_ratio:
            figname = f'fig_entsoePs_ratio_{cc}'
            ylms = (-1,42.0,)
            plt.plot_date(drange,100*(dps_ - dps0)/np.mean(dps0),'k-',lw=0.7,)
            # plt.title(
            #  f'{cc}, Recon. Error : {rerr(dps0,dps_,p=0,ord=1,):.2%}')
                # f'{cc}, RMS relative difference: {rerr(dps0,dps_,p=0,):.2%}')
            plt.ylabel("$\dfrac{ \mathrm{Max\,outage} - \mathrm{Min\,outage} }{ \mathrm{Overall\;mean\;outage} } $ , %")
            plt.xticks(rotation=30)
        
        plt.xlabel('Datetime')
        plt.xlim((drange[0],drange[-1]))
        for yr in range(min(drange).year,max(drange).year+1):
            plt.fill_between(
                [datetime(yr,11,eomf.get_nov_day(yr)) + timedelta(td) 
                    for td in [0,20*7]],[ylms[0]]*2,[ylms[1]]*2,
                                        color='k',alpha=0.1,)
        
        ylm = plt.ylim(ylms)
        if sf:
            sff(figname,sd_mod='fig_entsoePs',sd=fig_sd,pdf=True,)
            plt.close()
        else:
            tlps()

if pltOneWeekXmpl or pltTsXmplRuns:
    ccsel = 'GB'
    fssel = (3.0,2.8) if pltOneWeekXmpl else (2.6,2.8)
    
    fig, ax = plt.subplots(figsize=fssel)
    drange,_,_,dpsXr = eomf.load_dps(ds,de,cc,sd,rerun=False)
    ndys = 7
    month = 2
    for yr in range(2016,2021):
        i0 = np.argmax(drange==datetime(yr,month,1,))
        plt.plot(np.arange(24*ndys)/24,dpsXr['t'][i0:i0+24*ndys]/1e3,
                                                            label=str(yr))
    
    ylm = plt.ylim()
    ylms = (-0.2,ylm[1],)
    xlms = (-0.05,ndys,)
    if pltOneWeekXmpl:
        plt.xlim(xlms)
        plt.ylim(ylms)
        plt.xticks(np.arange(8),)
        plt.xlabel(r'Days since 1st Feb',)
        plt.ylabel(f'{cc} Total Outage, GW',)
        plt.legend(title='Year',fontsize='small',handlelength=0.8,
                                                        loc=(1.03,0.35,))
        if sf:
            sff(f'pltOneWeekXmpl',sd=fig_sd,pdf=True,)
        
        tlps()
    else:
        plt.close()
    
    if pltTsXmplRuns:
        nt = 24*ndys
        fig, ax = plt.subplots(figsize=fssel,)
        for seed in 1000*np.arange(3):
            ua = self.build_unavl_model(assign=False,seed=seed,nt=nt,)
            outages = ua[ccsel]['v'].dot(ua[ccsel]['ua'])/1e3
            plt.plot(np.arange(len(outages))/(24),outages,)
        
        plt.xlabel('Time, days')
        plt.ylabel(f'{ccsel} Modelled Outages, GW')
        ylm = plt.ylim()
        plt.xticks(np.arange(8),)
        plt.xlim(xlms)
        plt.ylim(ylms)
        if sf:
            sff(f'pltTsXmplRuns',sd=fig_sd,pdf=True,)
        
        tlps()



if fig_entsoeout:
    # Plotting reports from individual days
    isel = {'GB':150,'IE':200}[cc]
    kk = kks[isel]
    mmRdr = list(APs[kk].keys())

    fig,ax = plt.subplots(figsize=(6,2.0 + len(mmRdr)*0.13))
    for i,m in enumerate(mmRdr):
        for aps in APs[kk][m]:
            # print(m)
            clr = 'r' if aps['changes'] else 'k'
            for ap in aps['data']:
                plt.plot_date([ap['start'],ap['end'],],[i,i],
                                                clr+'.-',alpha=0.2)

    plt.plot(np.nan,np.nan,'r.-',alpha=0.2,label='Forced')
    plt.plot(np.nan,np.nan,'k.-',alpha=0.2,label='Planned')
    plt.legend(title='Outage Type',fontsize='small')

    lbls = [mm[m]['psrName'] for m in mmRdr]
    plt.yticks(np.arange(len(mmRdr)),labels=lbls)
    ylm = plt.ylim()
    plt.vlines([eomf.s2d(kks[isel]),eomf.s2d(kks[isel+1])],*ylm,
                                                linestyles='dashed')
    plt.xlim((eomf.s2d(kks[isel-3]),eomf.s2d(kks[isel+4]),))
    plt.ylim(ylm)
    plt.xlabel('Time, UTC')
    plt.ylabel('Resource mRID')
    plt.xticks(rotation=30)
    if sf:
        sff(f'fig_entsoeout_{cc}_{kk}',sd=fig_sd,pdf=True,)
    
    tlps()


if fig_pngUnits or fig_hydro:
    # for options, see pprint(PSRTYPE_MAPPINGS)
    sd_ = os.path.join(sd,'png',cc,)
    P = {}
    
    _ = plt.subplots(figsize=(7,4)) if fig_pngUnits else None
    
    clrI = 0
    for kk,pmap in PSRTYPE_MAPPINGS.items():
        clrI+=1
        i_pmap, p_pmap, dd = mtList(3)
        for yr in range(2015,2021):
            for mo in range(1,13):
                dd.append(datetime(yr,mo,1))
                head,data = csvIn(os.path.join(sd_,f'png_{cc}_{eomf.m2s(dd[-1])}.csv'))
                
                idxPT,idxP = [head.index(h) for h in ['psrType','nomP']]
                i_pmap.append([i for i,d in enumerate(data) 
                                                    if (d[idxPT]==kk)])
                p_pmap.append([float(data[i][idxP]) for i in i_pmap[-1]])
        
        P[pmap] = [sum(p) for p in p_pmap]
        mrkr = '.-' if clrI<20 else 'x-'
        if sum(P[pmap])!=0 and fig_pngUnits:
            plt.plot_date(dd,P[pmap],
                            mrkr,label=pmap,color=cm.tab20(clrI % 20))
    
    if fig_pngUnits:
        plt.xlabel('Datetime')
        plt.ylabel(f'{cc} Installed Generation, MW')
        plt.legend(title='PSRType',loc=(1.05,0.05),fontsize='small',)
        if sf:
            sff(f'fig_pngUnits_{cc}')
        
        tlps()
    
    if fig_hydro:
        for kk in ['B11','B10']:
            i_pmap.append([i for i,d in enumerate(data) 
                                                if (d[idxPT]==kk)])
            p_pmap.append([float(data[i][idxP]) for i in i_pmap[-1]])
            dhh = [data[i] for i in i_pmap[-1]]
            
            dhh = pd.DataFrame(data=data,columns=head)
            dhyrdo = dhh[dhh['psrType']==kk]
            isel = np.argsort(dhyrdo['nomP'].astype(float)).to_numpy()
            yval = dhyrdo['nomP'].astype(float).iloc[isel].to_numpy()
            xval = np.arange(len(yval))
            ss = dhyrdo['mRID'].iloc[isel].to_list()
            
            fig, ax = plt.subplots(figsize=(9,4))
            plt.plot(xval,yval,'.-',)
            for x,y,s in zip(xval,yval,ss):
                plt.text(x,y,s[9:],rotation=60)
            
            plt.title(
                    f'{cc} {PSRTYPE_MAPPINGS[kk]}, sum: {sum(yval):.0f} MW')
            plt.xlabel('Generator number (sorted)')
            plt.ylabel('Generator nominal power, MW')
            plt.ylim((-10,1150))
            tl()
            if sf:
                sff(f'plt_hydro_{cc}_{PSRTYPE_MAPPINGS[kk]}_{eomf.m2s(dd[-1])}')
            
            tlps()


if pltTsGenerator:
    bog = eomf.bzOutageGenerator()
    
    # transition probability - ccgt
    st_avl = 0.989
    lt_avl = 0.9
    
    # From 
    lmd = 1-st_avl
    mu = lmd*lt_avl/(1-lt_avl)
    
    n_yrs = 1
    ng = 100
    nt = n_yrs*20*7*24
    
    fig, axs = plt.subplots(figsize=(7.3,6.5),nrows=3,sharey=True,)
    for ax,ng in zip(axs,[10,30,100]):
        avl = bog.build_unavl_matrix(lmd,mu,lt_avl,ng,nt,)
        asd = np.ones(ng).dot(avl)
        
        ax.plot(np.arange(len(asd))/(24*7),100*asd/ng,
                                            label='$N_{\mathrm{gen}}$='+f'{ng}')
        ax.legend()
        ax.set_xlabel('Week of the winter')
        ax.set_ylabel('% total generation\nout of service')
    
    if sf:
        sff('pltTsGenerator')
    
    tlps()
    
    out_avl = 1 - (np.sum(avl)/(ng*avl.shape[1]))
    print(out_avl)
    
    state0 = [np.where(aa==0)[0][:-1] for aa in avl]
    trn_mean = [sum(avl[i,aa+1])/len(aa) for i,aa in enumerate(asd)]
    st_avl_out = 1 - np.mean(trn_mean)
    print(st_avl_out)

if pltTsCcWinters:
    ua = self.build_unavl_model(assign=False,seed=0,)
    for cc in ua.ccs:
        # Calculate the outages and long-term mean outage rate, in GW
        outages = ua[cc]['v'].dot(ua[cc]['ua'])/1e3
        isel = np.where(np.not_equal(np.nansum(ua[cc]['ua'],axis=1),0))[0]
        lt_mean = np.nansum(ua[cc]['v'][isel] - ua[cc]['v_lt'][isel])/1e3
        
        fig, ax = plt.subplots(figsize=(5.5,3.2),)
        plt.plot(np.arange(len(outages))/(24*7),outages,)
        plt.xlabel('Time, weeks')
        plt.ylabel('Outages, GW')
        plt.title(f'{cc} Outages, long-term mean: {lt_mean:.2f} GW')
        if sf:
            sff(f'pltTsCcWinters_{cc}',sd_mod='pltTsCcWinters',)
            plt.close()
        else:
            tlps()

if plt_approach_xmpl_vvmults or plt_approach_xmpl_contiguous:
    cc_ = 'FR'
    APs, mm, kks, kksD = eomf.load_aps(cc_,sd,rerun=rerun,save=True)
    
    ds = datetime(2017,8,28,)
    de = datetime(2017,9,2,)
    _ = eomf.load_dps(ds,de,cc_,sd,rerun=1,save=1,)
    drange, moX, moXx, mo2x, mo2x_real = eomf.load_mouts(ds,de,cc_,sd,)
    
    if plt_approach_xmpl_vvmults:
        figname = 'plt_approach_xmpl_vvmults'
        k = '17W100P100P00788'
    elif plt_approach_xmpl_contiguous:
        figname = 'plt_approach_xmpl_contiguous'
        k = '17W100P100P0018Q'
    
    fig, ax = plt.subplots(figsize=(9,3.2))
    plt.plot_date(drange,mo2x(k),'.-',)
    plt.title(f"{cc_}, Generator: {mm[k]['psrName']} ({k})")
    plt.xlabel('Datetime (UTC)',)
    plt.ylabel('Capacity reduction, MW')
    plt.xlim((ds,de))
    if sf:
        sff(figname)
    
    tlps()
    
    # plt_approach_xmpl_vvmults Entsoe:
    # https://transparency.entsoe.eu/outage-domain/r2/unavailabilityOfProductionAndGenerationUnits/show?name=&defaultValue=false&viewType=TABLE&areaType=CTA&atch=false&dateTime.dateTime=26.08.2017+00:00|UTC|DAY&dateTime.endDateTime=03.09.2017+00:00|UTC|DAY&CTY|10YFR-RTE------C|MULTI=CTY|10YFR-RTE------C|MULTI&area.values=CTY|10YFR-RTE------C!CTA|10YFR-RTE------C&assetType.values=PU&assetType.values=GU&outageType.values=A54&outageType.values=A53&outageStatus.values=A05&masterDataFilterName=&masterDataFilterCode=17W100P100P00788&dv-datatable_length=10
    

    # plt_approach_xmpl_contiguous Entsoe:
    # https://transparency.entsoe.eu/outage-domain/r2/unavailabilityOfProductionAndGenerationUnits/show?name=&defaultValue=false&viewType=TABLE&areaType=CTA&atch=false&dateTime.dateTime=26.08.2017+00:00|UTC|DAY&dateTime.endDateTime=01.09.2017+00:00|UTC|DAY&area.values=CTY|10YFR-RTE------C!CTA|10YFR-RTE------C&assetType.values=PU&assetType.values=GU&outageType.values=A54&outageType.values=A53&outageStatus.values=A05&masterDataFilterName=CORDEMAIS+5&masterDataFilterCode=&dv-datatable_length=10
    
    
    
    # Interesting note: there DOES seem to be no changes in GB planned outages 
    # between 22nd and 26th Dec2017.
    # https://transparency.entsoe.eu/outage-domain/r2/unavailabilityOfProductionAndGenerationUnits/show?name=&defaultValue=false&viewType=TABLE&areaType=CTA&atch=false&dateTime.dateTime=22.12.2017+00:00|UTC|DAY&dateTime.endDateTime=26.12.2017+00:00|UTC|DAY&CTY|10Y1001A1001A92E|MULTI=CTY|10Y1001A1001A92E|MULTI&area.values=CTY|10Y1001A1001A92E!CTA|10Y1001A1001A016&area.values=CTY|10Y1001A1001A92E!CTA|10YGB----------A&assetType.values=PU&assetType.values=GU&outageType.values=A53&outageStatus.values=A05&masterDataFilterName=&masterDataFilterCode=&dv-datatable_length=10
    


if plt_xmpl_1 or plt_xmpl_2 or plt_xmpl_3 or plt_xmpl_4 \
    or plt_xmpl_5 or plt_xmpl_n:
    
    cc_ = 'GB'
    APs, mm, kks, kksD = eomf.load_aps(cc_,sd,rerun=rerun,save=True)
    n_out = 3 # 3 for planned / forced / total; 2 for planned & forced.
    minmax = False # also if not updated then plot both the min and max vals
    rotn = 30 # xticks rotation
    fs_xmpl = (9,3.2)
    ms = 'o-'
    
    if plt_xmpl_1:
        xlm = (datetime(2017,2,5,),datetime(2017,2,24,),)
        k = 'COTPS-2'
        figname = 'plt_xmpl_1'
    
    if plt_xmpl_2:
        xlm = (datetime(2017,2,15,),datetime(2017,3,1,),)
        k = 'HEYM28'
        figname = 'plt_xmpl_2'
    
    if plt_xmpl_3:
        # xlm = (datetime(2016,10,1,),datetime(2016,10,10,),)
        xlm = (datetime(2016,10,4,),datetime(2016,10,12,),)
        k = 'FIDL-2'
        figname = 'plt_xmpl_3'
        fs_xmpl = fs_sngl_
        ms = '-'
    
    if plt_xmpl_4:
        xlm = (datetime(2019,1,8,),datetime(2019,3,28,),)
        k = 'DRAXX-4'
        figname = 'plt_xmpl_4'
        fs_xmpl = fs_sngl_
        ms = '-'
        n_out = 2
    
    if plt_xmpl_5:
        xlm = (datetime(2021,1,25,),datetime(2021,1,31,),)
        ms = '-'
        k = 'SCCL-2'
        fs_xmpl = fs_sngl_
        figname = 'plt_xmpl_5'
    
    if plt_xmpl_n:
        xlm = (datetime(2021,1,25,),datetime(2021,1,31,),)
        k = _CHANGEME_
        figname = _CHANGEME_
        n_out = 3 # up to 3
        minmax = BINARY
    
    # If needed, this version can re-run the analysis for a shorter period
    ds,de = xlm
    _ = eomf.load_dps(ds,de,cc_,sd,rerun=1,save=1,)
    drange, moX, moXx, mo2x, mo2x_real = eomf.load_mouts(ds,de,cc_,sd,)

    # # Essentially what we are plotting:
    # plt.plot_date(drange,mo2x(psrn2id[k])[:,:3],'.-',mfc='None',)
    # plt.plot_date(drange,mo2x(psrn2id[k])[:,3:],'.-',mfc='None',)
    
    fig, ax = plt.subplots(figsize=fs_xmpl,)
    _ = [plt.plot_date(drange,mo2x_real(psrn2id[k])[:,ii],ms,mfc='None',
                label=lbl,ms=2*(1+ii),) 
                    for ii,lbl in enumerate(['Plnd.','Frcd.','Ttl.',][:n_out])]
    
    if minmax:
        _ = [plt.plot_date(drange,mo2x(psrn2id[k])[:,3+ii],'o-',mfc='None',
                label=lbl,ms=2*(1+ii),) 
                    for ii,lbl in enumerate(['FrcdX','PlanX','TotX',][:n_out] )]
    
    plt.title(f"ID: {k} ({psrn2id[k]})",fontsize='medium',)
    plt.xlabel('Time, UTC',)
    plt.ylabel('Capacity reduction, MW')
    plt.legend(title='Outage Type',fontsize='small',loc=(1.02,0.25,),)
    nomP = float(mm[psrn2id[k]]['nomP'])
    plt.hlines(nomP,*xlm,'k',ls='--',lw=1.0,zorder=10)
    plt.text(xlm[0]+np.diff(xlm)*1.014,0.92*nomP,'Nominal\nCapacity',
                                                    fontsize='small',)
    plt.xlim(xlm)
    plt.yticks(np.arange(0,(1 + np.ceil(nomP/100))*100,100))
    plt.xticks(rotation=rotn,)
    if sf:
        sff(figname,sd=fig_sd,pdf=True,)

    tlps()

    # # # Note: using the outage generator requires data to be downloaded
    # # # as described in the readme.
    # # self = eomf.bzOutageGenerator(av_model=av_model,)

    # COTPS-2 February 2017 (plt_xmpl_1)
    # https://transparency.entsoe.eu/outage-domain/r2/unavailabilityOfProductionAndGenerationUnits/show?name=&defaultValue=false&viewType=TABLE&areaType=CTA&atch=false&dateTime.dateTime=01.02.2017+00:00|UTC|DAY&dateTime.endDateTime=28.02.2017+00:00|UTC|DAY&CTY|10Y1001A1001A92E|MULTI=CTY|10Y1001A1001A92E|MULTI&area.values=CTY|10Y1001A1001A92E!CTA|10Y1001A1001A016&area.values=CTY|10Y1001A1001A92E!CTA|10YGB----------A&assetType.values=PU&assetType.values=GU&outageType.values=A54&outageType.values=A53&outageStatus.values=A05&masterDataFilterName=COTPS-2&masterDataFilterCode=&dv-datatable_length=10

    # HEYM28 February 2017 (plt_xmpl_2)
    # https://transparency.entsoe.eu/outage-domain/r2/unavailabilityOfProductionAndGenerationUnits/show?name=&defaultValue=false&viewType=TABLE&areaType=CTA&atch=false&dateTime.dateTime=01.02.2017+00:00|UTC|DAY&dateTime.endDateTime=28.02.2017+00:00|UTC|DAY&area.values=CTY|10Y1001A1001A92E!CTA|10Y1001A1001A016&area.values=CTY|10Y1001A1001A92E!CTA|10YGB----------A&assetType.values=PU&assetType.values=GU&outageType.values=A54&outageType.values=A53&outageStatus.values=A05&masterDataFilterName=HEYM28&masterDataFilterCode=&dv-datatable_length=10

    # FIDL-2 Oct 2016 (plt_xmpl_3)
    # https://transparency.entsoe.eu/outage-domain/r2/unavailabilityOfProductionAndGenerationUnits/show?name=&defaultValue=false&viewType=TABLE&areaType=CTA&atch=false&dateTime.dateTime=01.10.2016+00:00|UTC|DAY&dateTime.endDateTime=31.10.2016+00:00|UTC|DAY&CTY|10Y1001A1001A92E|MULTI=CTY|10Y1001A1001A92E|MULTI&area.values=CTY|10Y1001A1001A92E!CTA|10Y1001A1001A016&area.values=CTY|10Y1001A1001A92E!CTA|10YGB----------A&assetType.values=PU&assetType.values=GU&outageType.values=A54&outageType.values=A53&outageStatus.values=A05&masterDataFilterName=FIDL-2&masterDataFilterCode=&dv-datatable_length=10

    # DRAXX-4 Spring 2019 (plt_xmpl_4)
    # https://transparency.entsoe.eu/outage-domain/r2/unavailabilityOfProductionAndGenerationUnits/show?name=&defaultValue=false&viewType=TABLE&areaType=CTA&atch=false&dateTime.dateTime=05.01.2019+00:00|UTC|DAY&dateTime.endDateTime=20.03.2019+00:00|UTC|DAY&CTY|10Y1001A1001A92E|MULTI=CTY|10Y1001A1001A92E|MULTI&area.values=CTY|10Y1001A1001A92E!CTA|10Y1001A1001A016&area.values=CTY|10Y1001A1001A92E!CTA|10YGB----------A&assetType.values=PU&assetType.values=GU&outageType.values=A54&outageType.values=A53&outageStatus.values=A05&masterDataFilterName=DRAXX-4&masterDataFilterCode=&dv-datatable_length=10

    # SCCL-2 Jan 2021 (plt_xmpl_5)
    # https://transparency.entsoe.eu/outage-domain/r2/unavailabilityOfProductionAndGenerationUnits/show?name=&defaultValue=false&viewType=TABLE&areaType=CTA&atch=false&dateTime.dateTime=26.01.2021+00:00|UTC|DAY&dateTime.endDateTime=31.01.2021+00:00|UTC|DAY&CTY|10Y1001A1001A92E|MULTI=CTY|10Y1001A1001A92E|MULTI&area.values=CTY|10Y1001A1001A92E!CTA|10Y1001A1001A016&area.values=CTY|10Y1001A1001A92E!CTA|10YGB----------A&assetType.values=PU&assetType.values=GU&outageType.values=A54&outageType.values=A53&outageStatus.values=A05&masterDataFilterName=SCCL-2&masterDataFilterCode=&dv-datatable_length=10


if tbl_fuel_mttrs:
    # table of availability, MTTR
    kks = list(eomf.ecr_tidy.keys())
    fleet_yr = self.fleets['GB'][2020]
    mttrs,mttfs = [{} for i in range(2)]
    rows = []
    k_s = []
    for k,v in fleet_yr.items():
        # Calculate the transition probabilities
        # Billinton + Allan Engineering Systems, 9.2.1
        k_ = self.cnvtr[k]
        k__ = self.cnvtr_ecr2elexon[k_]
        lmd = 1-self.st_avl[k__]
        mu = lmd*self.avlbty[k_]/(1-self.avlbty[k_])
        mttfs[k] = 1/lmd
        mttrs[k] = 1/mu
        if k_ in kks and k_ not in k_s and not k__ is None:
            rows.append([eomf.ecr_tidy[k_],
                                f'{self.avlbty[k_]:.2f}',f'{mttrs[k]:.1f}',])
            k_s.append(k_)
    
    caption = 'Availability factors and mean time to repair (MTTR)\\\\by fuel type.'
    label='tbl_fuel_mttrs'
    heading=['Fuel type','Availability $A$','MTTR']
    data = rows
    TD = tbl_sd
    basicTblSgn(caption,label,heading,data,TD,)



if tbl_acf:
    data = []
    jsel = [1,6,24,24*7]
    dr2 = np.arange(datetime(2000,11,1),datetime(2021,4,1),timedelta(0,3600,))
    ua = self.build_unavl_model(assign=False,seed=0,nt=len(dr2),)
    for cc_ in ['GB','FR','DE','ES','NL',]:
        dr1,dpsX,dpsXx,dpsXr = eomf.load_dps(ds,de,cc_,sd,rerun=False)
        mdl = ua[cc_]['v'].dot(ua[cc_]['ua'])
        
        dpssAll = []
        for xx,dr in zip([dpsXr['f'],dpsXr['t'],mdl,],[dr1,dr1,dr2],):
            dpssAll.append([self.getWinters(sctXt(t=dr,x=xx),yrStt=yr,nYr=1,) 
                                                for yr in range(2000,2021)] )
        
        acfs = [npa([acf(dpss[-5+ii].x,nlags=max(jsel),) for ii in range(5)]) 
                                                    for dpss in dpssAll]
        acfs_mn = npa([np.mean(acfx,axis=0) for acfx in acfs])
        
        data.append([cc_]+[f'{acfs_mn[ii,jj]:.2f}' for jj in jsel for ii in [1,2]])
    
    
    caption = 'Comparing the mean of the autocorrelation of Total outages in Western European counries for five winters (16/17 - 20/21) against models, considering lags from one hour to one week.'
    label = 'tbl_acf'
    heading = ['']*9
    TD = tbl_sd
    headRpl = r"""\multirow{3}*{\vspace{-1em} \begin{tabular}[x]{@{}r@{}}Ctry.\\Code\end{tabular} } & \multicolumn{8}{c}{Autocorrelation at given lag} \\
        \cmidrule(l{0.6em}r{0.9em}){2-9}
        & \multicolumn{2}{c}{1 hr} & \multicolumn{2}{c}{6 hrs} & \multicolumn{2}{c}{1 day} & \multicolumn{2}{c}{1 week}\\
        \cmidrule(l{0.6em}r{0.9em}){2-3} \cmidrule(l{0.6em}r{0.9em}){4-5} \cmidrule(l{0.6em}r{0.9em}){6-7} \cmidrule(l{0.6em}r{0.9em}){8-9} 
    & Data & Mdl. & Data & Mdl. & Data & Mdl. & Data & Mdl.\\"""
    basicTblSgn(caption,label,heading,data,TD,headRpl,r0=True,)
