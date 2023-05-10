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
from Python.pybold_master.pybold.hrf_model import spm_hrf
from Python.pybold_master.pybold.bold_signal import deconv
#from Python.python_scripts.wauwterfmri import gloverhrf

def bold_deconvolve(timeseries,hrf=None,TR=1,nb_iter=100):
    if hrf is None:
        hrf_dur = 30
        true_hrf_delta = 1.5
        hrf, t_hrf = spm_hrf(t_r=TR, delta=true_hrf_delta, dur=hrf_dur)
    
    params = {'y': timeseries,
              't_r': TR,
              'hrf': hrf,
              'lbda': None,
              'nb_iter': nb_iter,
              'verbose': 1,
              }
    
    est_ar_s, est_ai_s, est_i_s, J, R, G = deconv(**params)
    return est_ar_s, est_ai_s, est_i_s

def gppi(designmatrix,seedtimeseries,contrasts=None):
    
    # hrf=gloverhrf(30,1)
    # est_ar_s, est_ai_s, est_i_s = bold_deconvolve(seedtimeseries,hrf=hrf)
    
    hrf0, t_hrf = spm_hrf(t_r=1, delta=1.5, dur=len(seedtimeseries))
    hrf, t_hrf = spm_hrf(t_r=1, delta=1.5, dur=30)
    
    sts=copy.deepcopy(seedtimeseries)
    sts[np.abs(sts)<np.std(sts)]=0
    
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
        ppi[ppi>0.15]=1
        ppi-=0.5
        ppi*=contrasts[i]
        #ppi*=est_ai_s
        ppi*=decon
        ppic=np.convolve(ppi,hrf)
        ppi_mat[i,:]=ppic[0:designmatrix.shape[1]]
        
    return ppi_mat