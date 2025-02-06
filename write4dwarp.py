#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:54:35 2024

@author: WauWter
"""
#%% Initialize
import os
os.chdir('/project/3017081.01/required/')
import numpy as np
from Python.python_scripts.wauwternifti import readnii,savenii
import sys

#%% Read layfile

if (len(sys.argv) == 5):
    nii_x=sys.argv[1]
    nii_y=sys.argv[2]
    nii_z=sys.argv[3]
    nii_o=sys.argv[4]
    
    niix,hdr=readnii(nii_x)
    niiy,hdr=readnii(nii_y)
    niiz,hdr=readnii(nii_z)
    
    niiw=np.zeros(niix.shape+(4,),dtype=np.float32)
    niiw[:,:,:,:,0]=np.flip(niix,axis=1)
    del niix
    niiw[:,:,:,:,1]=np.flip(niiy,axis=1)
    del niiy
    niiw[:,:,:,:,2]=np.flip(niiz,axis=1)
    del niiz
    
    hdr['dim']=(5,)+hdr['dim'][1:5]+(4,)+hdr['dim'][6:]
    hdr['pixdim']=hdr['pixdim'][:4]+(1.0,)+hdr['pixdim'][5:]
    hdr['datatype']=16
    hdr['bitpix']=32
    hdr['vox_offset']=352
    
    savenii(niiw,hdr,nii_o)
    
elif (len(sys.argv) == 4):
    nii_i=sys.argv[1]
    nii_o=sys.argv[2]
    nr=int(sys.argv[3])
    
    nii,hdr=readnii(nii_i)
    
    niiw=np.zeros((nii.shape[0],nii.shape[1],nii.shape[2],nr,4),dtype=np.float32)
    niiw[:,:,:,:,0]=np.reshape(np.repeat(nii[:,:,:,0],nr),(nii.shape[0],nii.shape[1],nii.shape[2],nr))
    niiw[:,:,:,:,1]=np.reshape(np.repeat(nii[:,:,:,1],nr),(nii.shape[0],nii.shape[1],nii.shape[2],nr))
    niiw[:,:,:,:,2]=np.reshape(np.repeat(nii[:,:,:,2],nr),(nii.shape[0],nii.shape[1],nii.shape[2],nr))
    
    del nii
    
    hdr['dim']=(5,)+hdr['dim'][1:4]+(nr,4,)+hdr['dim'][6:]
    hdr['pixdim']=hdr['pixdim'][:4]+(1.0,)+hdr['pixdim'][5:]
    hdr['datatype']=16
    hdr['bitpix']=32
    hdr['vox_offset']=352
    
    savenii(niiw,hdr,nii_o)
    