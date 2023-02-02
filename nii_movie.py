#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 01:47:03 2023

@author: WauWter
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Python.python_scripts.wauwternifti import readnii

def nii_movie(niifile,slicex=None,slicey=None,slicez=None,windowheight=6,colmap='gray',speed=10,scale=20):
    if isinstance(niifile,str):
        nii,hdr=readnii(niifile)
    else:
        nii=niifile
        
    if scale !=100:
        nii[nii>np.max(nii)*(scale/100)]=np.max(nii)*(scale/100)
    
    niisize=nii.shape
    global sx, sy, sz, frame, frl
    if slicex==None:
        sx=niisize[0]//2
    if slicey==None:
        sy=niisize[1]//2
    if slicez==None:
        sz=niisize[2]//2
    frl=niisize[3]
    frame=0
    
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(2*windowheight, windowheight))
    imx=ax1.imshow(nii[:,:,sz,frame],cmap=colmap)
    imy=ax2.imshow(nii[:,sy,:,frame],cmap=colmap)
    imz=ax3.imshow(nii[sx,:,:,frame],cmap=colmap)

    def update(*args):
        global sx, sy, sz, frame, frl
     
        imx.set_array(nii[:,:,sz,frame])
        imy.set_array(nii[:,sy,:,frame])
        imz.set_array(nii[sx,:,:,frame])

        frame += 1
        frame %= frl

        return imx,imy,imz

    ani = animation.FuncAnimation(fig, update, interval=speed)
    plt.show()
    return ani