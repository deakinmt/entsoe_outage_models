import os, sys, pickle, types, zipfile
import pandas as pd

from timeit import default_timer as timer

with open(os.path.join(sys.path[0],'import_list.py')) as file:
    exec(file.read())

# Config parameters
from download_config import dstart, dend, dstart_xtra, dend_xtra, dT
ds, de = dstart_xtra, dend_xtra
from scipy import sparse
from eom_utils import aMulBsp

# Select the countries to use
ccs = ['GB', 'IE', 'I0', 'BE', 'NL', 'FR', 'DK', 'ES', 'NO', 'DE',]

# Process the data to create the 
process_APs = 0
process_DPs = 0
save_outputs = 0

# Plotting options
fig_entsoeout = 0
fig_entsoePs = 0
fig_entsoePs_minmax = 0
fig_entsoePs_ratio = 0
fig_pngUnits = 0
fig_hydro = 0

# Misc figures
pltTsGenerator = 0
pltTsCcWinters = 0
pltOutageChanges = 0

# Save figure flag
sf = 0

# Script options
rerun = 0
cc = 'GB' # opts - 'GB', 'IE', 'I0', 'BE', 'NL', 'FR', 'ES', 'DK', 'DE',
av_model = 'ecr'

# Load the data and get the clock & keys
APs, mm, kks, kksD = eomf.load_aps(cc,sd,rerun=rerun,save=True)
self = eomf.bzOutageGenerator(av_model=av_model,)


if pltOutageChanges:
    drange,dpsF,dpsP,dpsFx,dpsPx = eomf.load_dps(ds,de,'GB',sd,rerun=False)
    for xx,nm in zip([dpsF,dpsF + dpsP],['Forced','Total',]):
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

if process_APs:
    for cc in ccs:
        APs, mm, kks, kksD = eomf.load_aps(cc,sd,rerun=True,save=True)

if process_DPs:
    for cc in ccs:
        drange,dpsF,dpsP,dpsFx,dpsPx = eomf.load_dps(
                                    ds,de,cc,sd,rerun=True,save=True)

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
"""

if save_outputs:
    dn_out = os.path.join(fn_root,'data_out',)
    with open(os.path.join(dn_out,'readme.txt'),'w') as file:
        file.write(save_readme)
    
    for cc in ccs:
        drange,dpsF,dpsP,dpsFx,dpsPx = eomf.load_dps(
                                    ds,de,cc,sd,rerun=False,save=False)
        fn = os.path.join(dn_out,f'outages_{cc}',)
        
        data = np.concatenate([
            [dpsF+(dpsFx*0.5),],
            [dpsP+(dpsPx*0.5),],
            [dpsFx*0.5,],
            [dpsPx*0.5,],
        ]).T
        dataHead = [
            'forced_outages_mw',
            'planned_outages_mw',
            'forced_outages_uncertainty_mw',
            'planned_outages_uncertainty_mw',
        ]
        saveDataFunc(fn,mode='csv',data=data,dataHead=dataHead,tStamps=drange,)


if fig_entsoePs or fig_entsoePs_minmax or fig_entsoePs_ratio:
    for cc in ccs:
        drange,dpsF,dpsP,dpsFx,dpsPx = eomf.load_dps(ds,de,cc,sd,rerun=False)
        dps = dpsF + dpsP
        dps_ = dpsF + dpsP + dpsFx + dpsPx
        
        fig, ax = plt.subplots(figsize=(9,3.2))
        if fig_entsoePs or fig_entsoePs_minmax:
            ylms = (-500,max(dps)*1.15)
            if fig_entsoePs:
                figname = f'fig_entsoePs_{cc}'
                plt.plot_date(drange,dpsP,'-',label='Planned',lw=0.7,)
                plt.plot_date(drange,dpsF,'-',label='Forced',lw=0.7,)
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
            plt.plot_date(drange,100*(dps_ - dps)/np.mean(dps),'k-',lw=0.7,)
            plt.title(
                f'{cc}, RMS relative difference: {rerr(dps,dps_,p=0,):.2%}')
            plt.ylabel("Relative difference: $\dfrac{ \mathrm{Max\,outage} - \mathrm{Min\,outage} }{ \mathrm{Mean\,outage} } $ , %")
        
        plt.xlabel('Datetime')
        plt.xlim((drange[0],drange[-1]))
        for yr in range(min(drange).year,max(drange).year+1):
            plt.fill_between(
                [datetime(yr,11,eomf.get_nov_day(yr)) + timedelta(td) 
                    for td in [0,20*7]],[ylms[0]]*2,[ylms[1]]*2,
                                        color='k',alpha=0.1,)
        
        ylm = plt.ylim(ylms)
        if sf:
            sff(figname,sd_mod='fig_entsoePs',)
            plt.close()
        else:
            tlps()


if fig_entsoeout:
    # Plotting reports from individual days
    isel = 150
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
    plt.title(
        f'Reporting Window: {eomf.s2d(kks[isel])} - {eomf.s2d(kks[isel+1])}')
    ylm = plt.ylim()
    plt.vlines([eomf.s2d(kks[isel]),eomf.s2d(kks[isel+1])],*ylm,
                                                linestyles='dashed')
    plt.xlim((eomf.s2d(kks[isel-4]),eomf.s2d(kks[isel+5]),))
    plt.ylim(ylm)
    plt.xlabel('Datetime')
    plt.ylabel('Resource mRID')
    plt.xticks(rotation=30)
    if sf:
        sff(f'fig_entsoeout_{cc}_{kk}')

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
    trn_avl = 0.989
    lt_avl = 0.9
    
    n_yrs = 1
    ng = 100
    nt = n_yrs*20*7*24

    fig, axs = plt.subplots(figsize=(7.3,6.5),nrows=3,sharey=True,)
    for ax,ng in zip(axs,[10,30,100]):
        avl = bog.build_unavl_matrix(trn_avl,lt_avl,ng,nt,)
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
