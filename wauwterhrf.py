#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:42:15 2023

@author: Wauwter
"""

import numpy as np
from scipy.special import erf
from Python.python_scripts.wauwterfmri import doublegammahrf


def param_convolve(hrf,par0):
    onset=par0[0]*par0[4]
    durat=par0[1]*par0[4]
    xax=np.arange(len(hrf)*3)
    act=(erf(xax-(onset+len(hrf)))+1)/2
    deact=(1-erf(xax-(onset+durat+len(hrf))))/2
    desmat=act*deact
    condes=np.convolve(desmat,hrf)
    condes=condes[len(hrf):2*len(hrf)]
    condes/=np.max(condes)
    condes*=par0[3]
    condes+=par0[2]
    return condes
    
def deconvolve(params,hrf,ydata):
    par0=np.zeros(5,dtype=np.float32)
    par0[0]=params['onset'].value
    par0[1]=params['duration'].value
    par0[2]=params['const'].value
    par0[3]=params['amplitude'].value
    par0[4]=params['Hz'].value
    ymodel=param_convolve(hrf,par0)
    return (ymodel-ydata)

def lmfit_doublegamma(params,hrflen,ydata,timestep):
    p0=np.zeros(7,dtype=np.float32)
    p0[0]=params['tpeak'].value
    p0[1]=params['tunder'].value
    p0[2]=params['dpeak'].value
    p0[3]=params['dunder'].value
    p0[4]=params['pu_ratio'].value
    p0[5]=params['const'].value
    p0[6]=params['amplitude'].value
    ymodel=doublegammahrf(p0,hrflen,timestep)
    return (ymodel-ydata)    
