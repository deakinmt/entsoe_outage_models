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
ccs = ['GB', 'IE', 'I0', 'BE', 'NL', 'FR']

# Process the data to create the 
process_APs = 0
process_DPs = 0

# Plotting options
fig_entsoeout = 0
fig_entsoePs = 0
fig_pngUnits = 0
fig_hydro = 0

# Misc figures
pltTsGenerator = 0

# Save figure flag
sf = 0

# Script options
rerun = 0
cc = 'GB' # opts - 'GB', 'IE', 'I0', 'BE', 'NL', 'FR'

# Load the data and get the clock & keys
APs, mm, kks, kksD = eomf.load_aps(cc,sd,rerun=rerun,save=True)

if process_APs:
    for cc in ccs:
        APs, mm, kks, kksD = eomf.load_aps(cc,sd,rerun=True,save=True)

if process_DPs:
    for cc in ccs:
        drange, dpsF, dpsP = eomf.load_dps(
                                    ds,de,cc,sd,rerun=True,save=True)

if fig_entsoePs:
    for cc in ccs:
        drange,dpsF,dpsP = eomf.load_dps(ds,de,cc,sd,rerun=False)
        dps = dpsF + dpsP
        
        nanPlot = np.nan*np.ones(len(drange))
        nanPlot[np.where(np.isnan(dps))[0]] = 0
        ylms = (-500,max(dps)*1.15)
        
        fig, ax = plt.subplots(figsize=(9,3.2))
        plt.plot_date(drange,dpsP,'-',label='Planned',)
        plt.plot_date(drange,dpsF,'-',label='Forced',)
        plt.plot_date(drange,dps,'k--',label='Total',)
        plt.plot_date(drange,nanPlot,'r.',ms=3)
        plt.legend()
        plt.xlabel('Datetime')
        plt.ylabel('Outages, MW')
        plt.title(f'{cc}; No. nans: {sum(np.isnan(dps))} oo {len(dps)}')
        plt.xlim((drange[0],drange[-1]))
        for yr in range(min(drange).year,max(drange).year+1):
            plt.vlines([datetime(yr,11,eomf.get_nov_day(yr)) + timedelta(td) 
                                for td in [0,20*7]],*ylms,linestyles='dashed')
        
        ylm = plt.ylim(ylms)
        plt.ylim()
        tl()
        if sf:
            sff(f'fig_entsoePs_{cc}',sd_mod='fig_entsoePs',)
        
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
        avl, ttf, ttr = bog.build_avail_matrix(trn_avl,lt_avl,ng,nt,)
        asd = aMulBsp(np.ones(ng),avl)
        
        ax.plot(np.arange(len(asd))/(24*7),100*asd/ng,
                                            label='$N_{\mathrm{gen}}$='+f'{ng}')
        ax.legend()
        # ax.set_title(f'No. generators: {ng}')
        ax.set_xlabel('Week of the winter')
        ax.set_ylabel('% total generation\nout of service')
    
    if sf:
        sff('pltTsGenerator')
    
    tlps()

    out_avl = 1 - (np.sum(avl)/(ng*avl.shape[1]))
    print(out_avl)


