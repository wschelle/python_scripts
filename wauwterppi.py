#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:46:35 2023

@author: wousch
"""
import numpy as np
import copy
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d
#from Python.pybold_master.pybold.hrf_model import spm_hrf
#from Python.pybold_master.pybold.bold_signal import deconv
from Python.python_scripts.wauwterfmri import gammahrf

# def bold_deconvolve(timeseries,hrf=None,TR=1,nb_iter=100):
#     if hrf is None:
#         hrf_dur = 30
#         true_hrf_delta = 1.5
#         hrf, t_hrf = spm_hrf(t_r=TR, delta=true_hrf_delta, dur=hrf_dur)
    
#     params = {'y': timeseries,
#               't_r': TR,
#               'hrf': hrf,
#               'lbda': None,
#               'nb_iter': nb_iter,
#               'verbose': 1,
#               }
    
#     est_ar_s, est_ai_s, est_i_s, J, R, G = deconv(**params)
#     return est_ar_s, est_ai_s, est_i_s

def gppi(designmatrix,seedtimeseries,TR=1,cc_threshold=1,dm_threshold=0.15,contrasts=None):
    
    #hrf0, t_hrf = spm_hrf(t_r=1, delta=1.5, dur=len(seedtimeseries))
    #hrf, t_hrf = spm_hrf(t_r=1, delta=1.5, dur=30)
    hrf0 = gammahrf(len(seedtimeseries),TR)
    hrf = gammahrf(30,TR)
    
    sts=copy.deepcopy(seedtimeseries)
    sts[np.abs(sts)<(np.std(sts)*cc_threshold)]=0
    
    decon=correlate(hrf0,sts)
    decon=decon[0:len(seedtimeseries)]
    decon/=np.max(decon)
    decon=np.flip(decon)
    
    if contrasts==None:
        contrasts=np.ones(designmatrix.shape[0])
        
    ppi_mat=np.zeros(designmatrix.shape,dtype=np.float32)
    
    for i in range(designmatrix.shape[0]):
        ppi=copy.deepcopy(designmatrix[i,:])
        ppi=gaussian_filter1d(ppi,2,mode='mirror')
        ppi-=np.min(ppi)
        ppi/=np.max(ppi)
        ppi[ppi>dm_threshold]=1
        ppi-=0.5
        ppi*=contrasts[i]
        ppi*=decon
        ppic=np.convolve(ppi,hrf)
        ppi_mat[i,:]=ppic[0:designmatrix.shape[1]]
        
    return ppi_mat

def gppi_c(designmatrix,seedtimeseries,contrastmatrix,TR=1,cc_threshold=1,dm_threshold=0.1):
    # requires the following:
    # 1. designmatrix of shape [nr_conditions, nr_timepoints]
    # 2. seedtimeseries of shape [nr_timepoints]
    # 3. contrastmatrix of shape (nr_contrasts, nr_conditions)

    hrf0 = gammahrf(len(seedtimeseries),TR)
    hrf = gammahrf(30,TR)
    
    sts=copy.deepcopy(seedtimeseries)
    sts[np.abs(sts)<(np.std(sts)*cc_threshold)]=0
    
    decon=correlate(hrf0,sts)
    decon=decon[0:len(seedtimeseries)]
    decon/=np.max(decon)
    decon=np.flip(decon)
    
    ppi_mat=np.zeros((contrastmatrix.shape[0],designmatrix.shape[1]),dtype=np.float32)
    for i in range(contrastmatrix.shape[0]):
        ppi=copy.deepcopy(designmatrix)
        for j in range(designmatrix.shape[0]):
            ppi[j,:]*=contrastmatrix[i,j]
        
        ppi=np.sum(ppi,axis=0)
        ppi=gaussian_filter1d(ppi,1.25,mode='reflect')
        ppi-=np.min(ppi)
        ppi/=np.max(ppi)
        ppi-=0.5
        ppi[ppi>dm_threshold]=1
        ppi[ppi<-dm_threshold]=-1
        ppi*=decon
        ppic=np.convolve(ppi,hrf)
        ppi_mat[i,:]=ppic[0:designmatrix.shape[1]]
        
    return ppi_mat