#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:36:30 2023

@author: WauwTer
"""

#import os
import sys
import numpy as np
#os.chdir('/home/control/wousch/')
#from Python.python_scripts.wauwternifti import readnii,savenii
from wauwternifti import readnii,savenii

bids=sys.argv[1]
sub=sys.argv[2]

file1=bids+sub+'/derivatives/pipe/anat/c1'+sub+'_T1w.nii.gz'
file2=bids+sub+'/derivatives/pipe/anat/c2'+sub+'_T1w.nii.gz'
file3=bids+sub+'/derivatives/pipe/anat/c3'+sub+'_T1w.nii.gz'

nii1,hdr1=readnii(file1)
nii2,hdr2=readnii(file2)
nii3,hdr3=readnii(file3)

mask=np.zeros(nii1.shape,dtype=np.int16)
mask[nii1>0]=1
mask[nii2>0]=1
mask[nii3>0]=1

hdr1['bitpix']=16
hdr1['datatype']=4
hdr1['vox_offset']=352
hdr1['scl_slope']=1
hdr1['scl_inter']=0

savenii(mask,hdr1,bids+sub+'/derivatives/pipe/anat/mask.nii.gz')

