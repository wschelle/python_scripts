#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 14:37:40 2023

@author: WauWter
"""
#%% Initialize
import os
os.chdir('/project/3017081.01/required/')
import numpy as np
from Python.python_scripts.wauwternifti import readnii,savenii
import sys

#%% Read layfile

layfile=sys.argv[1]
# sub='sub-014'
# rdir='/project/3017081.01/bids7T/'+sub+'/'
# ddir=rdir+'derivatives/'
# anatdir=ddir+'pipe/anat/'

# layfile=anatdir+'layers_equidist-0.5mm.nii.gz'

lay,hlay=readnii(layfile)

lay[lay==1]+=1
lay[lay==20]-=1
lay[lay>0]-=1
nlay=3
laycorr=np.max(lay)/nlay
lay=np.ceil(lay/laycorr).astype(np.int16)

#%% make separate layer files

laydeep=np.zeros(lay.shape,dtype=np.int16)
laymid=np.zeros(lay.shape,dtype=np.int16)
laytop=np.zeros(lay.shape,dtype=np.int16)

laydeep[lay==1]=1
laymid[lay==2]=1
laytop[lay==3]=1

hlay['datatype']=4
hlay['bitpix']=16

layfilebase,layfilesuffix=layfile.split('.nii.gz')
savenii(laydeep,hlay,layfilebase+'_deep.nii.gz')
savenii(laymid,hlay,layfilebase+'_mid.nii.gz')
savenii(laytop,hlay,layfilebase+'_top.nii.gz')
