#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 19:31:22 2024

@author: WauWter
"""
#%% Initialize
import os
os.chdir('/project/3017081.01/required/')
import numpy as np
from Python.python_scripts.wauwternifti import readnii,savenii
import sys

#%% Read niifile

nii_i=sys.argv[1]

nii,hdr=readnii(nii_i)

nii=nii.astype(np.float32)

hdr['dim']=(5,)+hdr['dim'][1:4]+(1,3,)+hdr['dim'][6:]
hdr['datatype']=16
hdr['bitpix']=32
hdr['vox_offset']=352

niifilebase,niifilesuffix=nii_i.split('.nii.gz')

for i in range(nii.shape[3]):
    savenii(nii[:,:,:,i,:3],hdr,niifilebase+'_'+f"{i:04d}"+'.nii.gz')