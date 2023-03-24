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


def wm_onsets_V2(csvfile,precision=1,firstart=-1,revert_precision=False):
    df=pd.read_csv(csvfile)

    sdur=1.
    pdur=2.
    mdur=0.5

    scon=df['SampleCon'][1:-1]
    pcon=df['ProbeCon'][1:-1]

    starttime=df['StartFix.started'][1]
    starts1=df['InputFig1.started'][1:-1]
    starts2=df['InputFig2.started'][1:-1]
    starts3=df['InputFig3.started'][1:-1]
    startp1=df['OutputFig1.started'][1:-1]
    startp2=df['OutputFig2.started'][1:-1]
    startp3=df['OutputFig3.started'][1:-1]
    startm=df['MatchFig1.started'][1:-1]

    maxwm=df['MaxWM'][1:-1]
    mcon=df['MatchCon'][1:-1]
    mtrue=df['MatchTrue'][1:-1]
    cats=['face','scene','tool']
    catsp=['f_probe','s_probe','t_probe']
    ncat=len(cats)

    scon=scon.to_numpy()
    scon = scon[~np.isnan(scon)]
    pcon=pcon.to_numpy()
    pcon = pcon[~np.isnan(pcon)]
    maxwm=maxwm.to_numpy()
    maxwm = maxwm[~np.isnan(maxwm)]
    mcon=mcon.to_numpy()
    mcon = mcon[~np.isnan(mcon)]   
    mtrue=mtrue.to_numpy()
    mtrue = mtrue[~np.isnan(mcon)]
    mcon[(mcon==1)&(mtrue==0)]=2
    maxwm[maxwm>3]=3

    ncons=np.max(scon)+1
    ntrial=len(scon)
    nmatch1=np.sum(mcon==1)
    nmatch2=np.sum(mcon==2)
    nmatch=nmatch1+nmatch2

    starts1=starts1.to_numpy()
    starts2=starts2.to_numpy()
    starts3=starts3.to_numpy()
    startp1=startp1.to_numpy()
    startp2=startp2.to_numpy()
    startp3=startp3.to_numpy()
    startm=startm.to_numpy()

    starts1-=starttime
    starts2-=starttime
    starts3-=starttime
    startp1-=starttime
    startp2-=starttime
    startp3-=starttime
    startm-=starttime

    starts1*=precision
    starts2*=precision
    starts3*=precision
    startp1*=precision
    startp2*=precision
    startp3*=precision
    startm*=precision

    firstart*=precision

    # 4 input & 4 output + 1 match conditions (base setup)
    f1=np.zeros([ntrial*2+nmatch,3],dtype=np.float32)
    f1[0:ntrial,0]=scon
    f1[np.where(scon==0),1]=np.round(starts1[np.where(scon==0)]+firstart)
    f1[np.where((scon>0)&(df['SampleBox1'][1:-1]==1)),1]=np.round(starts1[np.where((scon>0)&(df['SampleBox1'][1:-1]==1))]+firstart)
    f1[np.where((scon>0)&(df['SampleBox1'][1:-1]==0)&(df['SampleBox2'][1:-1]==1)),1]=np.round(starts2[np.where((scon>0)&(df['SampleBox1'][1:-1]==0)&(df['SampleBox2'][1:-1]==1))]+firstart)
    f1[np.where((scon>0)&(df['SampleBox1'][1:-1]==0)&(df['SampleBox2'][1:-1]==0)&(df['SampleBox3'][1:-1]==1)),1]=np.round(starts3[np.where((scon>0)&(df['SampleBox1'][1:-1]==0)&(df['SampleBox2'][1:-1]==0)&(df['SampleBox3'][1:-1]==1))]+firstart)
    f1[np.where(scon==0),2]=sdur*ncat
    f1[np.where(scon>0),2]=sdur*scon[np.where(scon>0)]
    f1[ntrial:2*ntrial,0]=pcon+ncons
    f1[ntrial:2*ntrial,1]=np.round(startp1+firstart)
    f1[ntrial:2*ntrial,2]=pdur*(pcon)
    f1[2*ntrial:2*ntrial+nmatch1,0]=2*ncons
    f1[2*ntrial:2*ntrial+nmatch1,1]=np.round(startm[mcon==1]+firstart)
    f1[2*ntrial:2*ntrial+nmatch1,2]=mdur
    f1[2*ntrial+nmatch1:2*ntrial+nmatch1+nmatch2,0]=2*ncons+1
    f1[2*ntrial+nmatch1:2*ntrial+nmatch1+nmatch2,1]=np.round(startm[mcon==2]+firstart)
    f1[2*ntrial+nmatch1:2*ntrial+nmatch1+nmatch2,2]=mdur
    f1[f1[:,2]==0,2]=pdur
    f1=f1[f1[:,1].argsort()]

    # categorical input & output conditions
    f2=np.zeros([])
    for i in range(ntrial):
        for j in enumerate(cats):
            if (j[1] in df['Sample1'][i+1]) & (df['SampleBox1'][i+1]==1):
                f2=np.append(f2,[j[0]+1,np.round(starts1[i]+firstart),sdur])
            if (j[1] in df['Sample2'][i+1]) & (df['SampleBox2'][i+1]==1):
                f2=np.append(f2,[j[0]+1,np.round(starts2[i]+firstart),sdur])
            if (j[1] in df['Sample3'][i+1]) & (df['SampleBox3'][i+1]==1):
                f2=np.append(f2,[j[0]+1,np.round(starts3[i]+firstart),sdur])
        for k in enumerate(catsp):
            if (k[1] in df['Probe1'][i+1]):
                f2=np.append(f2,[k[0]+5,np.round(startp1[i]+firstart),pdur])
            if (k[1] in df['Probe2'][i+1]):
                f2=np.append(f2,[k[0]+5,np.round(startp2[i]+firstart),pdur])
            if (k[1] in df['Probe3'][i+1]):
                f2=np.append(f2,[k[0]+5,np.round(startp3[i]+firstart),pdur])
        if scon[i]==0:
            f2=np.append(f2,[0,np.round(starts1[i]+firstart),sdur*ncat])
        if pcon[i]==0:
            f2=np.append(f2,[4,np.round(startp1[i]+firstart),pdur*ncat])
        if mcon[i]>0:
            f2=np.append(f2,[8,np.round(startm[i]+firstart),mdur])
            
    f2=np.reshape(f2[1:],[int((f2.shape[0]-1)/3),3])
    f2=f2[f2[:,1].argsort()]

    # categorical input maintenance + n-back output conditions
    f3=np.zeros([])
    for i in range(ntrial):
        for j in enumerate(cats):
            if (j[1] in df['Sample1'][i+1]) & (df['SampleBox1'][i+1]==0):
                f3=np.append(f3,[j[0]+1,np.round(starts1[i]+firstart),sdur])
            if (j[1] in df['Sample2'][i+1]) & (df['SampleBox2'][i+1]==0):
                f3=np.append(f3,[j[0]+1,np.round(starts2[i]+firstart),sdur])
            if (j[1] in df['Sample3'][i+1]) & (df['SampleBox3'][i+1]==0):
                f3=np.append(f3,[j[0]+1,np.round(starts3[i]+firstart),sdur])
        if scon[i] == 3:
            f3=np.append(f3,[0,np.round(starts1[i]+firstart),sdur*ncat])
        if pcon[i] == 0:
            f3=np.append(f3,[4,np.round(startp1[i]+firstart),pdur*ncat])
        if pcon[i] > 0:
            f3=np.append(f3,[maxwm[i]+5,np.round(startp1[i]+firstart),pcon[i]*pdur])
        if mcon[i]>0:
            f3=np.append(f3,[9,np.round(startm[i]+firstart),mdur])
    
    f3=np.reshape(f3[1:],[int((f3.shape[0]-1)/3),3])
    f3=f3[f3[:,1].argsort()]
    
    if revert_precision:
        f1[:,1]/=precision
        f2[:,1]/=precision
        f3[:,1]/=precision
    
    return(f1,f2,f3)
