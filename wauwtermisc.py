#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:44:41 2023

@author: wousch
"""
import numpy as np

def unique(list1):
    # initialize a null list
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return(unique_list)

def butterworth_filter_3D(xdim, ydim, zdim, cutoff, order):
    distarr=np.zeros([xdim,ydim,zdim])
    for k in range(zdim):
        for j in range(ydim):
            for i in range(xdim):
                distarr[i,j,k] = np.sqrt(((i-xdim/2))**2 + ((j-ydim/2))**2 + ((k-zdim/2))**2)
    bFilter = 1.0 / np.sqrt((1 + (distarr/cutoff)**(2*order)))
    bFilter = np.roll(bFilter, (int(np.ceil(xdim/2)), int(np.ceil(ydim/2)), int(np.ceil(zdim/2))),axis=(0,1,2))
    return(bFilter)

def gaussian_filter_3D(sigmas,kernel_size):
    kernel=np.zeros([kernel_size,kernel_size,kernel_size])
    center=kernel_size//2
    for i in range(kernel_size):
        for j in range(kernel_size):
            for k in range(kernel_size):
                kernel[i,j,k]=np.exp(-(((center-i)**2/(2*sigmas[0]**2)) + ((center-j)**2/(2*sigmas[1]**2)) + ((center-k)**2/(2*sigmas[2]**2))))
    kernel/=np.max(kernel)
    return(kernel)

def convol_nan_3D(data,kernel,fillnan=False):
    result=np.zeros(data.shape)
    ksx=kernel.shape[0]
    ksy=kernel.shape[1]
    ksz=kernel.shape[2]
    data2=np.empty([data.shape[0]+2*ksx,data.shape[1]+2*ksy,data.shape[2]+2*ksz])
    data2[:]=np.nan
    data2[ksx:-ksx,ksy:-ksy,ksz:-ksz]=data
    for i in range(ksx,data.shape[0]+ksx):
        for j in range(ksy,data.shape[1]+ksy):
            for k in range(ksz,data.shape[2]+ksz):
                if (~np.isnan(data2[i,j,k])) | fillnan:
                    tmp=data2[i-ksx//2:i+round(ksx/2),j-ksy//2:j+round(ksy/2),k-ksz//2:k+round(ksz/2)]*kernel
                    result[i-ksx,j-ksy,k-ksz]=np.nansum(tmp)/np.sum(kernel[~np.isnan(tmp)])
    return(result)
    
