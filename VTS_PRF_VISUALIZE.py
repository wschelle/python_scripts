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

roifile='/home/wouter/Fridge/Laminar/VTS/V9934/data/func/func1/BA.lh.nii.gz'
layfile='/home/wouter/Fridge/Laminar/VTS/V9934/data/func/func1/-lay-equidist.nii.gz'

rdir=roifile[:-28]
sub=roifile[-34:-29]
ddir=rdir+'derivatives/'
ndir=ddir+'nii/'
fdir=ddir+'figures/'

print('Make pretty picas with: \n')
print(sub)
print('\n')

prfc1,hdrc=readnii(ndir+'pRF_center_digit.nii.gz')
prfc2,hdrc=readnii(ndir+'pRF_center_digit+palmwrist.nii.gz')
prfc3,hdrc=readnii(ndir+'pRF_center_phalanx.nii.gz')
prfc4,hdrc=readnii(ndir+'pRF_center_phalanx+palmwrist.nii.gz')
prfs,hdrs=readnii(ndir+'pRF_sigma_mean.nii.gz')
prfs1,hdrc=readnii(ndir+'pRF_sigma_digit.nii.gz')
prfs2,hdrc=readnii(ndir+'pRF_sigma_phalanx.nii.gz')
lay,hdrl=readnii(layfile)
roi,hdrr=readnii(roifile)
fval,hdrf=readnii(ndir+'pRF_goodness-of-fit-F.nii.gz')

fx=hdrc['dim'][1];fy=hdrc['dim'][2];fz=hdrc['dim'][3]
nvox=fx*fy*fz
prfc1=np.reshape(prfc1,nvox)
prfc2=np.reshape(prfc2,nvox)
prfc3=np.reshape(prfc3,nvox)
prfc4=np.reshape(prfc4,nvox)
prfs=np.reshape(prfs,nvox)
prfs1=np.reshape(prfs1,nvox)
prfs2=np.reshape(prfs2,nvox)
prfs0=deepcopy(prfs)
prfs0[prfs>2]=2

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

ncol=500
color = plt.colormaps['jet_r'].resampled(ncol)

roitits=['BA1','BA3b','BA3a','BA4p','BA4a']
laytits=['Deep','Middle','Superficial']
for i in range(nroi):
    for j in range(nlayi):
        prfct=prfc1-0.5
        prfct/=np.max(prfct)
        prfcc=color(prfct)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(niic[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1),0], 
            niic[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1),1], 
            niic[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1),2], 
            c=prfcc[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1),:], 
            s=fval2[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1)],
            marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title(roitits[i]+' '+laytits[j]+' layers')
        plt.show()
        angles = np.linspace(0,360,121)[:-1]
        rotanimate(ax, angles,fdir+'pRFcenter-digits-'+roitits[i]+'-'+laytits[j]+'.gif',delay=6,width=5,heigth=5)
        
        prfct=deepcopy(prfc3)
        prfct/=np.max(prfct)
        prfcc=color(prfct)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(niic[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1),0], 
            niic[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1),1], 
            niic[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1),2], 
            c=prfcc[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1),:], 
            s=fval2[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1)],
            marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title(roitits[i]+' '+laytits[j]+' layers')
        plt.show()
        angles = np.linspace(0,360,121)[:-1]
        rotanimate(ax, angles,fdir+'pRFcenter-phalanx-'+roitits[i]+'-'+laytits[j]+'.gif',delay=6,width=5,heigth=5) 
        
        prfct=deepcopy(prfs0)
        prfct/=np.max(prfct)
        prfcc=color(prfct)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(niic[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1),0], 
           niic[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1),1], 
           niic[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1),2], 
           c=prfcc[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1),:], 
           s=fval2[(roi==i+1)&(layi==j+1)&(fval>fth)&(coormask[:,i]==1)],
           marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title(roitits[i]+' '+laytits[j]+' layers')
        plt.show()
        angles = np.linspace(0,360,121)[:-1]
        rotanimate(ax, angles,fdir+'pRFsigma-'+roitits[i]+'-'+laytits[j]+'.gif',delay=6,width=5,heigth=5) 

