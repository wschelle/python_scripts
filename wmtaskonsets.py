#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 14:37:43 2023

@author: wousch
"""
import pandas as pd
import numpy as np
import copy

def wm_onsets(csvfile,precision=1,firstart=-1):
    df=pd.read_csv(csvfile)

    scon=df['SampleCon']
    pcon=df['ProbeCon']
    starttime=df['FixCrossBlank.started']
    startsample=df['InputFixCross.started']
    startprobe=df['OutputFixCross.started']
    maxwm=df['MaxWM']
    mcon=df['MatchCon']
    cats=['face','scene','tool']

    scon=scon.to_numpy()
    scon = scon[~np.isnan(scon)]
    pcon=pcon.to_numpy()
    pcon = pcon[~np.isnan(pcon)]
    maxwm=maxwm.to_numpy()
    maxwm = maxwm[~np.isnan(maxwm)]
    mcon=mcon.to_numpy()
    mcon = mcon[~np.isnan(mcon)]   

    ncons=np.max(scon)+1
    ntrial=len(scon)

    st=starttime.to_numpy()
    st, = st[~np.isnan(st)]
    ss=startsample.to_numpy()
    ss = ss[~np.isnan(ss)]
    sp=startprobe.to_numpy()
    sp = sp[~np.isnan(sp)]

    ss-=st
    sp-=st
    ss*=precision
    sp*=precision
    firstart*=precision

    # 4 input & 4 output conditions (base setup)
    f1=np.zeros([ntrial*2,2],dtype=int)
    f1[0:ntrial,0]=scon
    f1[0:ntrial,1]=np.round(ss+firstart)
    f1[ntrial:,0]=pcon+ncons
    f1[ntrial:,1]=np.round(sp+firstart)
    f1=f1[f1[:,1].argsort()]

    # categorical input & output conditions
    f2=np.zeros([])
    for i in range(ntrial):
        for j in enumerate(cats):
            if (j[1] in df['SampleLeft'][i+1]) & (df['SampleBoxLeft'][i+1]==1):
                f2=np.append(f2,[j[0]+1,np.round(ss[i]+firstart)])
            if (j[1] in df['SampleRight'][i+1]) & (df['SampleBoxRight'][i+1]==1):
                f2=np.append(f2,[j[0]+1,np.round(ss[i]+firstart)])
            if (j[1] in df['SampleTop'][i+1]) & (df['SampleBoxTop'][i+1]==1):
                f2=np.append(f2,[j[0]+1,np.round(ss[i]+firstart)])
            if (j[1] in df['ProbeLeft'][i+1]) & (df['ProbeBoxLeft'][i+1]==1):
                f2=np.append(f2,[j[0]+5,np.round(sp[i]+firstart)])
            if (j[1] in df['ProbeRight'][i+1]) & (df['ProbeBoxRight'][i+1]==1):
                f2=np.append(f2,[j[0]+5,np.round(sp[i]+firstart)])
            if (j[1] in df['ProbeTop'][i+1]) & (df['ProbeBoxTop'][i+1]==1):
                f2=np.append(f2,[j[0]+5,np.round(sp[i]+firstart)])
        if scon[i]==0:
            f2=np.append(f2,[0,np.round(ss[i]+firstart)])
        if pcon[i]==0:
            f2=np.append(f2,[4,np.round(sp[i]+firstart)])
            
    f2=np.reshape(f2[1:],[int((f2.shape[0]-1)/2),2])
    f2=f2[f2[:,1].argsort()].astype(int)

    # categorical input maintenance + n-back output conditions
    f3=np.zeros([])
    for i in range(ntrial):
        for j in enumerate(cats):
            if (j[1] in df['SampleLeft'][i+1]) & (df['SampleBoxLeft'][i+1]==0):
                f3=np.append(f3,[j[0]+1,np.round(ss[i]+firstart)])
            if (j[1] in df['SampleRight'][i+1]) & (df['SampleBoxRight'][i+1]==0):
                f3=np.append(f3,[j[0]+1,np.round(ss[i]+firstart)])
            if (j[1] in df['SampleTop'][i+1]) & (df['SampleBoxTop'][i+1]==0):
                f3=np.append(f3,[j[0]+1,np.round(ss[i]+firstart)])
        if scon[i] == 3:
            f3=np.append(f3,[0,np.round(ss[i]+firstart)])
        if pcon[i] == 0:
            f3=np.append(f3,[4,np.round(sp[i]+firstart)])
        if pcon[i] > 0:
            f3=np.append(f3,[maxwm[i]+5,np.round(sp[i]+firstart)])
    
       
    f3=np.reshape(f3[1:],[int((f3.shape[0]-1)/2),2])
    f3=f3[f3[:,1].argsort()].astype(int)
    
    # selectiveness input & matching output conditions (base setup)
    scon2=copy.deepcopy(scon)
    scon2[scon2==2]=1
    scon2[scon2==3]=2
    f4=np.zeros([ntrial*2,2],dtype=int)
    f4[0:ntrial,0]=scon2
    f4[0:ntrial,1]=np.round(ss+firstart)
    f4[ntrial:,0]=mcon+ncons-1
    f4[ntrial:,1]=np.round(sp+firstart)
    f4=f4[f4[:,1].argsort()]
    
    return(f1,f2,f3,f4)