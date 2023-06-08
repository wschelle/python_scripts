#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 15:05:41 2023

@author: WauWter
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# from Python.python_scripts.wauwternifti import readnii,getniicoor
# from Python.python_scripts.aniplot import *
from wauwternifti import readnii,getniicoor
from aniplot import *
from copy import deepcopy
from tqdm import tqdm

nroi=5
nlayi=3
nlay=6
fth=2.5

layfile=sys.argv[1]
roifile=sys.argv[2]

# roifile='/home/wouter/Fridge/Laminar/VTS/V9934/data/func/func1/BA.lh.nii.gz'
# layfile='/home/wouter/Fridge/Laminar/VTS/V9934/data/func/func1/-lay-equidist.nii.gz'

rdir=roifile[:-28]
sub=roifile[-34:-29]
ddir=rdir+'derivatives/'
ndir=ddir+'nii/'
fdir=ddir+'figures/'

print('Make pretty picas for: \n')
print(sub)
print('\n')

prfc1,hdrc=readnii(ndir+'pRF_center_digit.nii.gz')
prfc2,hdrc=readnii(ndir+'pRF_center_digit+palmwrist.nii.gz')
prfc3,hdrc=readnii(ndir+'pRF_center_phalanx.nii.gz')
prfc4,hdrc=readnii(ndir+'pRF_center_phalanx+palmwrist.nii.gz')
prfs,hdrs=readnii(ndir+'pRF_sigma_mean.nii.gz')
prfs1,hdrc=readnii(ndir+'pRF_sigma_digit.nii.gz')
prfs2,hdrc=readnii(ndir+'pRF_sigma_phalanx.nii.gz')
ampl,hdrc=readnii(ndir+'pRF_amplitude.nii.gz')
lay,hdrl=readnii(layfile)
roi,hdrr=readnii(roifile)
fval,hdrf=readnii(ndir+'pRF_goodness-of-fit-F.nii.gz')

fx=hdrc['dim'][1];fy=hdrc['dim'][2];fz=hdrc['dim'][3]
nvox=fx*fy*fz
prfc1=np.reshape(prfc1,nvox)
prfc2=np.reshape(prfc2,nvox)
prfc3=np.reshape(prfc3,nvox)
prfc4=np.reshape(prfc4,nvox)
ampl=np.reshape(ampl,nvox)
prfs=np.reshape(prfs,nvox)
prfs1=np.reshape(prfs1,nvox)
prfs2=np.reshape(prfs2,nvox)
hg,ed=np.histogram(prfs,bins=500,range=(0.01,prfs.max()))
hg2=deepcopy(hg)
for i in range(0,500):
    hg2[i]=np.sum(hg[0:i+1])/np.sum(hg)*100
ed2=ed[:-1]
prfscut=np.max(ed2[hg2<=90])
prfs0=deepcopy(prfs)
prfs0[prfs>prfscut]=prfscut

lay2=deepcopy(lay)
lay2[lay==1]+=1
lay2[lay==20]-=1
lay2[lay>0]-=1
layi=np.ceil(lay2/6)
lay=np.ceil(lay2/3)
lay=np.reshape(lay,nvox)
layi=np.reshape(layi,nvox)
del lay2

roi=np.reshape(roi,nvox)
roi[roi==2]=0
roi[roi==4]=2
roi[roi==6]=4
roi[roi>6]=0

fval=np.reshape(fval,nvox)
fval2=deepcopy(fval)
fval2[fval>0]-=np.min(fval[fval>0])
fval2/=np.max(fval2)
fval2[fval>0]*=19
fval2[fval>0]+=1

niic=getniicoor('/home/wouter/Fridge/Laminar/VTS/V9934/nii/pRF_center_digit.nii.gz')
niic=np.reshape(niic,(nvox,3))

centcoor=np.zeros([nroi,3])
for i in range(nroi):
    centcoor[i,:]=np.mean(niic[(roi==i+1)&(lay>0)&(fval>fth),:],axis=0)

niic2=np.reshape(niic,[fx,fy,fz,3])
coormask=np.zeros([fx,fy,fz,nroi],dtype=np.int16)
maxradius=20 #mm
for i in tqdm(range(fx)):
    for j in range(fy):
        for k in range(fz):
            for l in range(nroi):
                if (niic2[i,j,k,0]>centcoor[l,0]-maxradius) & (niic2[i,j,k,0]<centcoor[l,0]+maxradius) & (niic2[i,j,k,1]>centcoor[l,1]-maxradius) & (niic2[i,j,k,1]<centcoor[l,1]+maxradius) & (niic2[i,j,k,2]>centcoor[l,2]-maxradius) & (niic2[i,j,k,2]<centcoor[l,2]+maxradius):
                    coormask[i,j,k,l]=1

coormask=np.reshape(coormask,[nvox,nroi])

roitits=['BA1','BA3b','BA3a','BA4p','BA4a']
laytits=['Deep','Middle','Superficial']
for i in range(nroi):
    for j in range(nlayi):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection='3d')
        p=ax.scatter(niic[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1),0], 
            niic[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1),1], 
            niic[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1),2], 
            c=prfc1[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1)], 
            s=fval2[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1)],
            marker='o',cmap=plt.cm.jet_r)
        ax.set_xlabel('RL')
        ax.set_ylabel('AP')
        ax.set_zlabel('FH')
        ax.set_title('pRFc_digit '+roitits[i]+' '+laytits[j]+' layers')
        fig.colorbar(p, ax=ax, shrink=0.65)
        plt.show()
        angles = np.linspace(0,360,121)[:-1]
        rotanimate(ax, angles,fdir+'pRFc-digits-'+roitits[i]+'-'+laytits[j]+'.gif',delay=6,width=5,heigth=5)
        
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection='3d')
        p=ax.scatter(niic[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1),0], 
            niic[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1),1], 
            niic[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1),2], 
            c=prfc3[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1)], 
            s=fval2[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1)],
            marker='o',cmap=plt.cm.jet_r)
        ax.set_xlabel('RL')
        ax.set_ylabel('AP')
        ax.set_zlabel('FH')
        ax.set_title('pRFc_phalanx '+roitits[i]+' '+laytits[j]+' layers')
        fig.colorbar(p, ax=ax, shrink=0.65)
        plt.show()
        angles = np.linspace(0,360,121)[:-1]
        rotanimate(ax, angles,fdir+'pRFc-phalanx-'+roitits[i]+'-'+laytits[j]+'.gif',delay=6,width=5,heigth=5) 
        
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection='3d')
        p=ax.scatter(niic[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1),0], 
           niic[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1),1], 
           niic[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1),2], 
           c=prfs0[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1)], 
           s=fval2[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1)],
           marker='o',cmap=plt.cm.jet_r)
        ax.set_xlabel('RL')
        ax.set_ylabel('AP')
        ax.set_zlabel('FH')
        ax.set_title('pRFs '+roitits[i]+' '+laytits[j]+' layers')
        fig.colorbar(p, ax=ax, shrink=0.65)
        plt.show()
        angles = np.linspace(0,360,121)[:-1]
        rotanimate(ax, angles,fdir+'pRFs-mean'+roitits[i]+'-'+laytits[j]+'.gif',delay=6,width=5,heigth=5) 


# Binning the shit outta this
ndig=5
ndig2=7
nphal=3
nphal2=5
phallab=['tip','mid','base']
diglab=['thumb','index','middle','ring','little','palm','wrist']

sigdig=np.zeros([ndig2,nphal2,nroi,nlay],dtype=np.float32)
sigdige=np.zeros([ndig2,nphal2,nroi,nlay],dtype=np.float32)
sigphal=np.zeros([ndig2,nphal2,nroi,nlay],dtype=np.float32)
sigphale=np.zeros([ndig2,nphal2,nroi,nlay],dtype=np.float32)
amplit=np.zeros([ndig2,nphal2,nroi,nlay],dtype=np.float32)
amplite=np.zeros([ndig2,nphal2,nroi,nlay],dtype=np.float32)

prfc2[prfc2==0.5]+=0.01 #small adjustment to get the bins right
prfc4[prfc4==0.5]+=0.01 #small adjustment to get the bins right

for i in tqdm(range(ndig2)):
    for j in range(nphal2):
        for k in range(nroi):
            for l in range(nlay):
                if np.sum((prfc2>i+0.5)&(prfc2<=i+1.5)&(prfc4>j+0.5)&(prfc4<=j+1.5)&(roi==k+1)&(lay==l+1)&(fval>fth)&(coormask[:,k]==1))>0:
                    sigdig[i,j,k,l]=np.mean(prfs1[(prfc2>i+0.5)&(prfc2<=i+1.5)&(prfc4>j+0.5)&(prfc4<=j+1.5)&(roi==k+1)&(lay==l+1)&(fval>fth)&(coormask[:,k]==1)])
                    sigdige[i,j,k,l]=np.std(prfs1[(prfc2>i+0.5)&(prfc2<=i+1.5)&(prfc4>j+0.5)&(prfc4<=j+1.5)&(roi==k+1)&(lay==l+1)&(fval>fth)&(coormask[:,k]==1)]) / np.sqrt(np.sum((prfc2>=i+0.5)&(prfc2<=i+1.5)&(prfc4>=j+0.5)&(prfc4<=j+1.5)&(roi==k+1)&(lay==l+1)&(fval>fth)&(coormask[:,k]==1)))
                    sigphal[i,j,k,l]=np.mean(prfs2[(prfc2>i+0.5)&(prfc2<=i+1.5)&(prfc4>j+0.5)&(prfc4<=j+1.5)&(roi==k+1)&(lay==l+1)&(fval>fth)&(coormask[:,k]==1)])
                    sigphale[i,j,k,l]=np.std(prfs2[(prfc2>i+0.5)&(prfc2<=i+1.5)&(prfc4>j+0.5)&(prfc4<=j+1.5)&(roi==k+1)&(lay==l+1)&(fval>fth)&(coormask[:,k]==1)]) / np.sqrt(np.sum((prfc2>=i+0.5)&(prfc2<=i+1.5)&(prfc4>=j+0.5)&(prfc4<=j+1.5)&(roi==k+1)&(lay==l+1)&(fval>fth)&(coormask[:,k]==1)))
                    amplit[i,j,k,l]=np.mean(ampl[(prfc2>i+0.5)&(prfc2<=i+1.5)&(prfc4>j+0.5)&(prfc4<=j+1.5)&(roi==k+1)&(lay==l+1)&(fval>fth)&(coormask[:,k]==1)])
                    amplite[i,j,k,l]=np.std(ampl[(prfc2>i+0.5)&(prfc2<=i+1.5)&(prfc4>j+0.5)&(prfc4<=j+1.5)&(roi==k+1)&(lay==l+1)&(fval>fth)&(coormask[:,k]==1)]) / np.sqrt(np.sum((prfc2>=i+0.5)&(prfc2<=i+1.5)&(prfc4>=j+0.5)&(prfc4<=j+1.5)&(roi==k+1)&(lay==l+1)&(fval>fth)&(coormask[:,k]==1)))
                else:
                    sigdig[i,j,k,l]=np.nan
                    sigdige[i,j,k,l]=np.nan
                    sigphal[i,j,k,l]=np.nan
                    sigphale[i,j,k,l]=np.nan
                    amplit[i,j,k,l]=np.nan
                    amplite[i,j,k,l]=np.nan
                    
sigdig[5,0,:,:]=sigdig[5,3,:,:]
sigdig[6,0,:,:]=sigdig[6,4,:,:]
sigdige[5,0,:,:]=sigdige[5,3,:,:]
sigdige[6,0,:,:]=sigdige[6,4,:,:]
sigdig2=np.nanmean(sigdig,axis=3)
sigdige2=np.nanmean(sigdige,axis=3)

sigphal[5,0,:,:]=sigphal[5,3,:,:]
sigphal[6,0,:,:]=sigphal[6,4,:,:]
sigphale[5,0,:,:]=sigphale[5,3,:,:]
sigphale[6,0,:,:]=sigphale[6,4,:,:]
sigphal2=np.nanmean(sigphal,axis=3)
sigphale2=np.nanmean(sigphale,axis=3)

amplit[5,0,:,:]=amplit[5,3,:,:]
amplit[6,0,:,:]=amplit[6,4,:,:]
amplite[5,0,:,:]=amplite[5,3,:,:]
amplite[6,0,:,:]=amplite[6,4,:,:]
amplit2=np.nanmean(amplit,axis=3)
amplite2=np.nanmean(amplite,axis=3)

sigdig3=np.nanmean(sigdig,axis=1)
sigdige3=np.nanmean(sigdige,axis=1)
sigphal3=np.nanmean(sigphal,axis=1)
sigphale3=np.nanmean(sigphale,axis=1)
amplit3=np.nanmean(amplit,axis=1)
amplite3=np.nanmean(amplite,axis=1)

xax=np.arange(ndig2)+1
fig, axes = plt.subplots(1, nroi, figsize=(15,5))
for i in range(nroi):
    for j in range(nphal):
        axes[i].plot(xax,sigdig2[:,j,i],'o',label=phallab[j])
        axes[i].fill_between(xax, sigdig2[:,j,i] - sigdige2[:,j,i], sigdig2[:,j,i] + sigdige2[:,j,i], alpha=0.2)
        axes[i].set_ylim(ymin=0.1,ymax=prfscut+0.2)
        axes[i].set_title(roitits[i])
        axes[i].set_xticks(xax,labels=diglab,rotation=40)
        axes[i].spines[['right', 'top']].set_visible(False)
    if i==2:
        axes[i].set_xlabel('pRF center (digits)')
    if i==0:
        axes[i].set_ylabel('pRF size between digits')
    if i==4:
        axes[i].legend()
plt.savefig(fdir+'pRFs-between-digit_vs_pRFc-digits.png',dpi=300)
plt.show()

xax=np.arange(ndig2)+1
fig, axes = plt.subplots(1, nroi, figsize=(15,5))
for i in range(nroi):
    for j in range(nphal):
        axes[i].plot(xax,sigphal2[:,j,i],'o',label=phallab[j])
        axes[i].fill_between(xax, sigphal2[:,j,i] - sigphale2[:,j,i], sigphal2[:,j,i] + sigphale2[:,j,i], alpha=0.2)
        axes[i].set_ylim(ymin=0.1,ymax=prfscut+0.2)
        axes[i].set_title(roitits[i])
        axes[i].set_xticks(xax,labels=diglab,rotation=40)
        axes[i].spines[['right', 'top']].set_visible(False)
    if i==2:
        axes[i].set_xlabel('pRF center (digits)')
    if i==0:
        axes[i].set_ylabel('pRF size within digits')
    if i==4:
        axes[i].legend()
plt.savefig(fdir+'pRFs-within-digit_vs_pRFc-digits.png',dpi=300)
plt.show()

xax=np.arange(ndig2)+1
fig, axes = plt.subplots(1, nroi, figsize=(15,5))
for i in range(nroi):
    for j in range(nphal):
        axes[i].plot(xax,amplit2[:,j,i],'o',label=phallab[j])
        axes[i].fill_between(xax, amplit2[:,j,i] - amplite2[:,j,i], amplit2[:,j,i] + amplite2[:,j,i], alpha=0.2)
        axes[i].set_ylim(ymin=0.5,ymax=np.nanmax(amplit2)+0.25)
        axes[i].set_title(roitits[i])
        axes[i].set_xticks(xax,labels=diglab,rotation=40)
        axes[i].spines[['right', 'top']].set_visible(False)
    if i==2:
        axes[i].set_xlabel('pRF center (digits)')
    if i==0:
        axes[i].set_ylabel('BOLD amplitude')
    if i==4:
        axes[i].legend()
plt.savefig(fdir+'BOLD-amplitude_vs_pRFc-digits.png',dpi=300)
plt.show()

xax=np.arange(nlay)+1
fig, axes = plt.subplots(1, nroi, figsize=(15,5))
for i in range(nroi):
    for j in range(ndig):
        axes[i].plot(xax,sigdig3[j,i,:],'o',label=diglab[j])
        axes[i].fill_between(xax, sigdig3[j,i,:] - sigdige3[j,i,:], sigdig3[j,i,:] + sigdige3[j,i,:], alpha=0.2)
        axes[i].set_ylim(ymin=0.1,ymax=prfscut+0.2)
        axes[i].set_title(roitits[i])
        axes[i].spines[['right', 'top']].set_visible(False)
    if i==2:
        axes[i].set_xlabel('WM <---> GM')
    if i==0:
        axes[i].set_ylabel('pRF size between digits')
    if i==4:
        axes[i].legend()
plt.savefig(fdir+'pRFs-between-digit_vs_layers.png',dpi=300)
plt.show()

xax=np.arange(nlay)+1
fig, axes = plt.subplots(1, nroi, figsize=(15,5))
for i in range(nroi):
    for j in range(ndig):
        axes[i].plot(xax,sigphal3[j,i,:],'o',label=diglab[j])
        axes[i].fill_between(xax, sigphal3[j,i,:] - sigphale3[j,i,:], sigphal3[j,i,:] + sigphale3[j,i,:], alpha=0.2)
        axes[i].set_ylim(ymin=0.1,ymax=prfscut+0.2)
        axes[i].set_title(roitits[i])
        axes[i].spines[['right', 'top']].set_visible(False)
    if i==2:
        axes[i].set_xlabel('WM <---> GM')
    if i==0:
        axes[i].set_ylabel('pRF size within digits')
    if i==4:
        axes[i].legend()
plt.savefig(fdir+'pRFs-within-digit_vs_layers.png',dpi=300)
plt.show()


xax=np.arange(nlay)+1
fig, axes = plt.subplots(1, nroi, figsize=(15,5))
for i in range(nroi):
    for j in range(ndig):
        axes[i].plot(xax,amplit3[j,i,:],'o',label=diglab[j])
        axes[i].fill_between(xax, amplit3[j,i,:] - amplite3[j,i,:], amplit3[j,i,:] + amplite3[j,i,:], alpha=0.2)
        axes[i].set_ylim(ymin=0.5,ymax=np.nanmax(amplit3)+0.25)
        axes[i].set_title(roitits[i])
        axes[i].spines[['right', 'top']].set_visible(False)
    if i==2:
        axes[i].set_xlabel('WM <---> GM')
    if i==0:
        axes[i].set_ylabel('BOLD amplitude')
    if i==4:
        axes[i].legend()
plt.savefig(fdir+'BOLD-amplitude_vs_layers.png',dpi=300)
plt.show()



