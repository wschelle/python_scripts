# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 23:25:52 2022

@author: WauWter
"""
import pandas as pd
import numpy as np
import copy
from scipy.interpolate import interp1d
from python_scripts.wauwterfmri import gloverhrf

def wmcsv_convolve(csvfile):
    df=pd.read_csv(csvfile)
    timestep=0.1
    TR=1.
    idur=2.5
    odur=1.5
    
    hrf=gloverhrf(25,timestep)

    scon=df['SampleCon']
    pcon=df['ProbeCon']
    starttime=df['FixCrossBlank.started']
    startsample=df['InputFixCross.started']
    endsample=df['InputBlankFixCross.started']
    startprobe=df['OutputFixCross.started']
    endprobe=df['OutputBlankFixCross.started']
    endtime=df['Bye.started']
    
    wmface=df['WMFace']
    wmscene=df['WMScene']
    wmtool=df['WMTool']
    matchc=df['MatchCon']
    maxwm=df['MaxWM']
    cats=['face','scene','tool']

    scon=scon.to_numpy()
    scon = scon[~np.isnan(scon)]
    pcon=pcon.to_numpy()
    pcon = pcon[~np.isnan(pcon)]
    
    ncons=np.max(scon)+1
    nconp=np.max(pcon)+1
    ncont=ncons+nconp
    ntrial=len(scon)

    st=starttime.to_numpy()
    st, = st[~np.isnan(st)]
    ss=startsample.to_numpy()
    ss = ss[~np.isnan(ss)]
    es=endsample.to_numpy()
    es = es[~np.isnan(es)]
    sp=startprobe.to_numpy()
    sp = sp[~np.isnan(sp)]
    ep=endprobe.to_numpy()
    ep = ep[~np.isnan(ep)]
    et=endtime.to_numpy()
    et, = et[~np.isnan(et)]
    
    # wmface=wmface.to_numpy()
    # wmface = wmface[~np.isnan(wmface)]
    # wmscene=wmscene.to_numpy()
    # wmscene = wmscene[~np.isnan(wmscene)]
    # wmtool=wmtool.to_numpy()
    # wmtool = wmtool[~np.isnan(wmtool)]
    # matchc=matchc.to_numpy()
    # matchc = matchc[~np.isnan(matchc)]
    maxwm=maxwm.to_numpy()
    maxwm = maxwm[~np.isnan(maxwm)]

    ss-=st
    es-=st
    sp-=st
    ep-=st
    et-=st

    explen=round(et)
    explen2=explen/timestep

    dm_con=np.zeros([int(explen2),int(ncont)])
    dmc_con=np.zeros([int(explen2)+len(hrf)-1,int(ncont)])
    dmci_con=np.zeros([int(explen/TR),int(ncont)])
    
    dm_cat=np.zeros([int(explen2),int(ncont)])
    dmc_cat=np.zeros([int(explen2)+len(hrf)-1,int(ncont)])
    dmci_cat=np.zeros([int(explen/TR),int(ncont)])
    
    dm_nbac=np.zeros([int(explen2),int(ncont)])
    dmc_nbac=np.zeros([int(explen2)+len(hrf)-1,int(ncont)])
    dmci_nbac=np.zeros([int(explen/TR),int(ncont)]) 
    

    realtime=np.arange(0,explen,timestep)
    scantime=np.arange(0,explen,TR)

    for i in range(ntrial):
        dm_con[(ss[i]/timestep).astype(int):((ss[i]/timestep)+(idur/timestep)).astype(int),int(scon[i])]=1
        dm_con[(sp[i]/timestep).astype(int):((sp[i]/timestep)+(odur/timestep)).astype(int),int(pcon[i]+ncons)]=1
        dm_nbac[(sp[i]/timestep).astype(int):((sp[i]/timestep)+(odur/timestep)).astype(int),int(maxwm[i]+ncons)]=1
        for j in enumerate(cats):
            if (j[1] in df['SampleLeft'][i+1]) & (df['SampleBoxLeft'][i+1]==1):
                dm_cat[(ss[i]/timestep).astype(int):((ss[i]/timestep)+(idur/timestep)).astype(int),j[0]+1]=1
            if (j[1] in df['SampleRight'][i+1]) & (df['SampleBoxRight'][i+1]==1):
                dm_cat[(ss[i]/timestep).astype(int):((ss[i]/timestep)+(idur/timestep)).astype(int),j[0]+1]=1
            if (j[1] in df['SampleTop'][i+1]) & (df['SampleBoxTop'][i+1]==1):
                dm_cat[(ss[i]/timestep).astype(int):((ss[i]/timestep)+(idur/timestep)).astype(int),j[0]+1]=1
            if (j[1] in df['ProbeLeft'][i+1]) & (df['ProbeBoxLeft'][i+1]==1):
                dm_cat[(ss[i]/timestep).astype(int):((ss[i]/timestep)+(idur/timestep)).astype(int),j[0]+5]=1
            if (j[1] in df['ProbeRight'][i+1]) & (df['ProbeBoxRight'][i+1]==1):
                dm_cat[(ss[i]/timestep).astype(int):((ss[i]/timestep)+(idur/timestep)).astype(int),j[0]+5]=1
            if (j[1] in df['ProbeTop'][i+1]) & (df['ProbeBoxTop'][i+1]==1):
                dm_cat[(ss[i]/timestep).astype(int):((ss[i]/timestep)+(idur/timestep)).astype(int),j[0]+5]=1

    dm_cat[:,[0,4]]=copy.deepcopy(dm_con[:,[0,4]])
    dm_nbac[:,0:4]=copy.deepcopy(dm_con[:,0:4])

    for i in range(int(ncont)):
        dmc_con[:,i] = np.convolve(dm_con[:,i], hrf)
        dmc_con[:,i]/=np.max(dmc_con[:,i])
        fcon = interp1d(realtime,dmc_con[0:int(explen2),i],kind='cubic')
        dmci_con[:,i] = fcon(scantime)
        dmc_cat[:,i] = np.convolve(dm_cat[:,i], hrf)
        dmc_cat[:,i]/=np.max(dmc_cat[:,i])
        fcat = interp1d(realtime,dmc_cat[0:int(explen2),i],kind='cubic')
        dmci_cat[:,i] = fcat(scantime)
        dmc_nbac[:,i] = np.convolve(dm_nbac[:,i], hrf)
        dmc_nbac[:,i]/=np.max(dmc_nbac[:,i])
        fnb = interp1d(realtime,dmc_nbac[0:int(explen2),i],kind='cubic')
        dmci_nbac[:,i] = fnb(scantime)
    return(dmci_con,dmci_cat,dmci_nbac)
