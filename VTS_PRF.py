#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 12:31:13 2023

@author: WauWter
"""

# rdir='/Fridge/users/wouter/Laminar/VTS/'
# sub='V9934'
# vtsfile=rdir+sub+'/log/design_rand_csv.txt'
# niifile=rdir+sub+'/data/func/func1/func1-prep-ls.nii.gz'
# layfile=rdir+sub+'/data/func/func1/-lay-equidist.nii.gz'
# roifile=rdir+sub+'/data/func/func1/BA.lh.nii.gz'   


import os
import sys

niifile=sys.argv[1]
layfile=sys.argv[2]
roifile=sys.argv[3]
vtsfile=sys.argv[4]

print('Starting VTS 2D pRF analysis with: \n')
print('timeseries: '+niifile)
print('layer mask: '+layfile)
print('ROI mask: '+roifile)
print('VTS onsets: '+vtsfile)

rdir=niifile[:-42]
sub=niifile[-42:-37]
ddir=rdir+sub+'/derivatives/'
ndir=ddir+'nii/'
fdir=ddir+'figures/'

if not os.path.exists(ddir):
    os.mkdir(ddir)
if not os.path.exists(ndir):
    os.mkdir(ndir)
if not os.path.exists(fdir):
    os.mkdir(fdir)

import numpy as np
# from Python.python_scripts.wauwternifti import readnii,savenii
# from Python.python_scripts.wauwterfmri import *
from wauwternifti import readnii,savenii
from wauwterfmri import *
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import lmfit

def vts_cc(onsets,data,tr):
    lags=np.arange(-3,2,0.01,dtype=np.float32)
    cc=np.zeros((len(lags),data.shape[0]),dtype=np.float32)
    for i in tqdm(range(len(lags))):
        ons=copy.deepcopy(onsets)
        ons[:,1]+=lags[i]
        dm=hrf_convolve(ons,data.shape[1]*tr,TR=tr,upsample_factor=100)
        for j in range(data.shape[0]):
            cc[i,j]=np.corrcoef(dm,data[j,:])[0][1]
    return cc

def vts_prf2d(niifile,vtsfile,layfile=None,roifile=None,motpar=True,lowthreshold=1):
    onsets=np.loadtxt(vtsfile,usecols=range(7),delimiter=',')
    taskonsets=onsets[onsets[:,1]!=17]
    nconds=int(np.max(taskonsets[:,1:3]))
    taskons=np.zeros([taskonsets.shape[0]*2,3],dtype=np.float32)
    taskons[0:taskonsets.shape[0],0]=taskonsets[:,1]
    taskons[0:taskonsets.shape[0],1]=taskonsets[:,0]
    taskons[0:taskonsets.shape[0],2]=taskonsets[:,5]*10
    taskons[taskonsets.shape[0]:,0]=taskonsets[:,2]
    taskons[taskonsets.shape[0]:,1]=taskonsets[:,0]
    taskons[taskonsets.shape[0]:,2]=taskonsets[:,5]*10
    tasksort=taskons[:,1].argsort()
    taskons=taskons[tasksort,:]
    taskons[:,0]-=1
    taskons[:,1]-=2 #manual adjustment timing...
    
    tmptxt=niifile.split('.')
    fdir=tmptxt[0].split('-')
    fdir=fdir[0]
    del tmptxt
    
    nii,hdr=readnii(niifile)
    TR=hdr['pixdim'][4]
    fx=nii.shape[0]
    fy=nii.shape[1]
    fz=nii.shape[2]
    nscans=nii.shape[3]
    nvox=fx*fy*fz
    nii=np.reshape(nii,(nvox,nscans))
    
    mask=np.zeros(nvox,dtype=np.int16)
    mask[np.mean(nii,axis=1)>lowthreshold]=1
    
    if layfile:
        lay,hdrlay=readnii(layfile)
        lay=np.reshape(lay,nvox)
        mask[lay==0]=0
    if roifile:
        roi,hdroi=readnii(roifile)
        roi=np.reshape(roi,nvox)
        mask[(roi==0)|(roi==2)|(roi>6)]=0

    
    cut=0.01
    print('high-pass filtering:')
    nii2=hpfilt(nii,TR,cut,addfilt=0,mask=mask,convperc=1,showfiltmat=False)
    
    print('temporal smoothing:')
    nii3=np.zeros(nii2.shape,dtype=np.float32)
    for i in tqdm(range(nvox)):
        if mask[i]!=0:
            nii3[i,:]=gaussian_filter(nii2[i,:],sigma=0.8,mode='mirror',radius=8)
    del nii,nii2
    
    if motpar:
        print('head motion correction:')
        mp=np.loadtxt(fdir+'-mp')
        mp2=hpfilt(mp.T,TR,cut,addfilt=0,mask=0,convperc=0,showfiltmat=False)
        mp3=np.zeros([mp2.shape[0],nscans],dtype=np.float32)
        mp4=np.zeros([mp2.shape[0]*2,nscans],dtype=np.float32)
        for i in range(1,nscans):
            mp3[:,i]=mp2[:,i]-mp2[:,i-1]
        for i in range(mp2.shape[0]):
            mp4[i,mp3[i,:]>0]=mp3[i,mp3[i,:]>0]
            mp4[i+mp2.shape[0],mp3[i,:]<0]=mp3[i,mp3[i,:]<0]
        mp5=np.concatenate((mp2,mp4),axis=0)  
        mp6=np.zeros(mp5.shape,dtype=np.float32)   
        for i in range(mp5.shape[0]):
            mp6[i,:]=gaussian_filter(mp5[i,:],sigma=0.8,mode='mirror',radius=8)
        del mp,mp2,mp3,mp4,mp5
        
        be,ms,yh=GLM(mp6,nii3,mask,betaresid_only=False)
        for i in range(nvox):
            if mask[i]!=0:
                nii3[i,:]-=yh[i,:]
    
    print('VTS timing correction:')
    bscan=8
    nrep=8
    dumbcounter=np.zeros(nconds,dtype=np.int16)
    nii4=np.zeros([nvox,nrep,nconds,bscan],dtype=np.float32)
    for i in tqdm(range(len(taskons[:,0]))):
        if np.round((taskons[i,1]-TR)/TR)+bscan<=nscans:
            nii4[mask!=0,dumbcounter[int(taskons[i,0])],int(taskons[i,0]),:]=nii3[mask!=0,int(np.round((taskons[i,1]-TR)/TR)):int(np.round((taskons[i,1]-TR)/TR))+bscan]
        else:
            tmp=np.round((taskons[i,1]-TR)/TR)+bscan-nscans
            nii4[mask!=0,dumbcounter[int(taskons[i,0])],int(taskons[i,0]),0:int(bscan-tmp)]=nii3[mask!=0,int(np.round((taskons[i,1]-TR)/TR)):]
        dumbcounter[int(taskons[i,0])]+=1

    nii4e=np.std(nii4,axis=1)
    nii4=np.mean(nii4,axis=1)
    nii4[nii4e>0]=nii4[nii4e>0]/nii4e[nii4e>0]

    nii5=np.max(nii4[:,:,0:4],axis=2)

    m2=np.zeros(nvox,dtype=np.int16)
    for i in tqdm(range(nvox)):
        if np.max(nii5[i,:]>2):
            m2[i]=np.where(nii5[i,:]==np.max(nii5[i,:]))[0][0]+1

    tmp=np.where((m2>0)&(lay>0)&((roi==4)|(roi==1)))[0]
    nii6=np.zeros([nvox,bscan],dtype=np.float32)
    for i in tqdm(range(nvox)):
        if m2[i]>0:
            nii6[i,:]=nii4[i,m2[i]-1,:]
            
    nii6=nii6[tmp,:]

    to=np.array([[0,TR,3.2]])
    cc=vts_cc(to,nii6,TR)
    cc2=np.mean(cc,axis=1)
    lags=np.arange(-3,2,0.01,dtype=np.float32)
    plt.plot(lags,cc2)
    print('lag: ',lags[cc2.argmax()])
    shi=lags[cc2.argmax()]

    taskons[:,1]+=shi
    design_matrix=hrf_convolve(taskons,nscans*TR,TR=TR,upsample_factor=100)
    del nii4,nii4e,dumbcounter,nii5,nii6,m2,to,cc,cc2,lags,tmp
            

    print('Linear regression:')
    beta,msres=GLM(design_matrix,nii3,mask)
    
    contrasts=np.zeros([nconds+1,nconds],dtype=np.float32)
    for i in range(nconds):
        contrasts[i,i]=1
    
    print('simple t-statistics:')
    tval=np.zeros([nvox,nconds],dtype=np.float32)
    for i in range(nconds):
        tval[:,i]=tcon(contrasts[:,i],design_matrix,beta,msres,mask)
    
    print('writing beta- and t-maps:')
    subdir=fdir[0:-21]
    for i in range(nconds):
        savenii(np.reshape(beta[:,i],(fx,fy,fz)),hdrlay,ndir+'beta-c'+str(i)+'.nii.gz')
        savenii(np.reshape(tval[:,i],(fx,fy,fz)),hdrlay,ndir+'tmap-c'+str(i)+'.nii.gz')
   
    mask2=np.zeros(nvox,dtype=np.int16)
    mask2[(mask==1)&(np.max(tval,axis=1)>2)]=1
    
    handdim=5
    handmap=np.zeros([handdim,handdim],dtype=np.int16)
    handmap[0,0]=1
    handmap[1,0]=2
    handmap[0,1]=3
    handmap[1,1]=4
    handmap[2,1]=5
    handmap[0,2]=6
    handmap[1,2]=7
    handmap[2,2]=8
    handmap[0,3]=9
    handmap[1,3]=10
    handmap[2,3]=11
    handmap[0,4]=12
    handmap[1,4]=13
    handmap[2,4]=14
    handmap[3,2]=15
    handmap[4,2]=16
    
    betahand=np.zeros([nvox,handdim,handdim],dtype=np.float32)
    tmaphand=np.zeros([nvox,handdim,handdim],dtype=np.float32)
    dmhand=np.zeros([handdim,handdim,nscans],dtype=np.float32)
    for i in range(handdim):
        for j in range(handdim):
            if handmap[i,j]!=0:
                betahand[:,i,j]=beta[:,handmap[i,j]-1]
                tmaphand[:,i,j]=tval[:,handmap[i,j]-1]
                dmhand[i,j,:]=design_matrix[handmap[i,j]-1,:]
            
    gest=np.zeros([nvox,6],dtype=np.float32)
    print('calculating pRF estimates')
    for i in tqdm(range(nvox)):
        if mask2[i]!=0:
            gest[i,0]=np.mean(nii3[i,:])
            gest[i,1]=np.std(nii3[i,:])
            topfac=np.where(tmaphand[i,:,:]==np.max(tmaphand[i,:,:]))
            gest[i,2]=topfac[0][0]
            gest[i,3]=topfac[1][0]
            gest[i,4]=0.75+(gest[i,3]/4)
            gest[i,5]=1.
    
    xgrid = np.arange(handdim,dtype=np.float32)
    ygrid = np.arange(handdim,dtype=np.float32)
    ygrid, xgrid = np.meshgrid(xgrid, ygrid)
    
    nii4=nii3[mask2==1,:]
    m2list=np.where(mask2==1)[0]
    m2vox=len(m2list)
    prf_zfit=np.zeros([nvox,7],dtype=np.float32)
    
    print('Precise calculation pRF:')
    for i in tqdm(range(m2vox)):
        params = lmfit.Parameters()
        params.add('cons', gest[m2list[i],0])
        params.add('amp', gest[m2list[i],1],min=0)
        params.add('centerX', gest[m2list[i],2],min=-0.5,max=4.5)
        params.add('centerY', gest[m2list[i],3],min=-0.5,max=4.5)
        params.add('XYsigma', gest[m2list[i],4],min=0.1,max=1.2*handdim)
        params.add('XYsigmaRatio', gest[m2list[i],5],min=0.5,max=1.5)
        zfit = lmfit.minimize(lmfit_prf2d, params, args=(dmhand, nii4[i,:], xgrid, ygrid))
        prf_zfit[m2list[i],0]=zfit.params['cons'].value
        prf_zfit[m2list[i],1]=zfit.params['amp'].value
        prf_zfit[m2list[i],2]=zfit.params['centerX'].value
        prf_zfit[m2list[i],3]=zfit.params['centerY'].value
        prf_zfit[m2list[i],4]=zfit.params['XYsigma'].value
        prf_zfit[m2list[i],5]=zfit.params['XYsigmaRatio'].value
        prf_zfit[m2list[i],6]=zfit.chisqr
    
    del nii4, m2vox, m2list
    
    npar=6
    dfn=(npar-1)-1
    dfd=nscans-(npar-1)-1
    prf_yfit=np.zeros(nii3.shape,dtype=np.float32)
    print('determine pRF model:')
    for i in tqdm(range(nvox)):
        if mask2[i]!=0:
            if prf_zfit[i,6]>0:
                prf_yfit[i,:]=prf2d(dmhand,np.squeeze(prf_zfit[i,0:6]),xgrid,ygrid)
    
    del xgrid, ygrid

    print('calculate f-statistic:')
    fval=np.zeros(nvox,dtype=np.float32)
    rval=np.zeros(nvox,dtype=np.float32)
    for i in tqdm(range(nvox)):
        if (np.std(prf_yfit[i,:]) > 0) & (np.sum(np.isnan(prf_zfit[i,:]))==0):
            fval[i]=(np.sum((np.mean(nii3[i,:])-prf_yfit[i,:])**2)/dfn) / (prf_zfit[i,6]/dfd)
            tmp=np.corrcoef(np.squeeze(prf_yfit[i,:]),np.squeeze(nii3[i,:]))
            rval[i]=tmp[0,1]
    
    fingers=prf_zfit[:,3]
    phalange=prf_zfit[:,2]
    fingers[(fval>0)]+=1
    phalange[(fval>0)]+=1
    
    
    fingers[(phalange>3.5)&(phalange<=4.5)]=6
    fingers[(phalange>4.5)]=7
    savenii(np.reshape(fingers,(fx,fy,fz)),hdrlay,ndir+'pRF_center_digit+palmwrist.nii.gz')
    fingers[fingers>5.5]=0
    savenii(np.reshape(fingers,(fx,fy,fz)),hdrlay,ndir+'pRF_center_digit.nii.gz')
            
    savenii(np.reshape(phalange,(fx,fy,fz)),hdrlay,ndir+'pRF_center_phalanx+palmwrist.nii.gz')
    phalange[phalange>3.5]=0
    savenii(np.reshape(phalange,(fx,fy,fz)),hdrlay,ndir+'pRF_center_phalanx.nii.gz')
  
    savenii(np.reshape(prf_zfit[:,1],(fx,fy,fz)),hdrlay,ndir+'pRF_amplitude.nii.gz')
    savenii(np.reshape(fval,(fx,fy,fz)),hdrlay,ndir+'pRF_goodness-of-fit-F.nii.gz')
    savenii(np.reshape(rval**2,(fx,fy,fz)),hdrlay,ndir+'pRF_Rsquared.nii.gz')
        
    sigmaX=prf_zfit[:,4]
    sigmaY=prf_zfit[:,4]/prf_zfit[:,5]
    sigmaY[np.isnan(sigmaY)]=0
    sigma=np.vstack((sigmaX,sigmaY)).T
    sigmaMax=np.max(sigma,axis=1)
    sigmaMean=np.mean(sigma,axis=1)

    savenii(np.reshape(sigmaX,(fx,fy,fz)),hdrlay,ndir+'pRF_sigma_digit.nii.gz')
    savenii(np.reshape(sigmaY,(fx,fy,fz)),hdrlay,ndir+'pRF_sigma_phalanx.nii.gz')
    savenii(np.reshape(sigmaMax,(fx,fy,fz)),hdrlay,ndir+'pRF_sigma_max.nii.gz')
    savenii(np.reshape(sigmaMean,(fx,fy,fz)),hdrlay,ndir+'pRF_sigma_mean.nii.gz')

     
vts_prf2d(niifile,vtsfile,layfile=layfile,roifile=roifile)
