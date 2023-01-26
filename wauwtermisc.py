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
