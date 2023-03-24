#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 13:21:40 2023

@author: WauWter
"""

import numpy as np

def gt_correlation_matrix(data_2d):
    nfactors=data_2d.shape[0]
    corrmatrix=np.zeros([nfactors,nfactors])
    for i in range(nfactors):
        for j in range(nfactors):
            corrmatrix[i,j]=np.corrcoef(data_2d[i,:],data_2d[j,:])[0][1]
    return(corrmatrix)

def gt_cormat_thresh(data_2d):
    cmat=gt_correlation_matrix(data_2d)
    nfactors=data_2d.shape[0]
    m=np.mean(cmat)
    sd=np.std(cmat)
    se=sd/np.sqrt(nfactors)
    threshold=m-1.96*se
    
    binary_cmat=np.zeros(cmat.shape,dtype=np.int16)
    binary_cmat[cmat > threshold]=1
    
    eye=np.eye(nfactors,dtype=np.int16)
    eye-=1
    eye*=-1
    bcmat=binary_cmat*eye
    
    tcmat=cmat*bcmat
    return(tcmat,bcmat)

def gt_wconn(tcmat):
    S=np.sum(tcmat,axis=1)
    return(S)

def gt_wclust(tcmat,bcmat):
    nfactors=tcmat.shape[0]
    
    Kn=0
    for i in range(3,nfactors):
        Kn+=(i-2)
    
    tri=np.zeros([nfactors,Kn,2],dtype=np.int16)
    for i in range(nfactors):
        if np.sum(bcmat[i,:]) >= 2:
            count=0
            tmploc=np.where(bcmat[i,:]==1)[0]
            for j in range(len(tmploc)):
                for k in range(len(tmploc)):
                    if k > j:
                        if bcmat[tmploc[j],tmploc[k]]==1:
                            tri[i,count,:]=[tmploc[j],tmploc[k]]
                            count+=1

    wgeotri=np.zeros([nfactors,Kn])
    for i in range(nfactors):
        for j in range(Kn):
            if np.sum(tri[i,j,:])>0:
                wgeotri[i,j]=(tcmat[i,tri[i,j,0]]*tcmat[i,tri[i,j,1]]*tcmat[tri[i,j,0],tri[i,j,1]])**(1/3)
    wgeotritot=np.sum(wgeotri,axis=1)/2
    
    nrtri=np.zeros(nfactors,dtype=np.int16)
    for i in range(nfactors):
        nrtri[i]=np.sum(np.sum(tri[i,:,:],axis=1) > 0)
    
    wcluscoef=np.zeros(wgeotritot.shape)
    for i in range(nfactors):
        if nrtri[i]>0:
            wcluscoef[i]=(2*wgeotritot[i])/(np.sum(bcmat[i,:])*(np.sum(bcmat[i,:])-1))
    
    return(wcluscoef)




















