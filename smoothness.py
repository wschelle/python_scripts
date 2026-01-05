#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 10:11:48 2025

@author: WauWter
"""
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_gradient_magnitude
from Python.python_scripts.wauwternifti import readnii

#just for here now
beta1,res1,yhat1=GLM(conv1,sm,m2,betaresid_only=False)
res1d=sm-yhat1
res4d=np.zeros((nv,ntt),dtype=np.float32)
res4d[mask==1,:]=res1d
res4d=np.reshape(res4d,(fx,fy,fz,ntt))
mask=np.reshape(mask,(fx,fy,fz))

def estimate_smoothness(res4d, mask, hdr):
    """
    Estimate smoothness (FWHM, dlh, resels) from residuals of a 4D GLM fit.
    
    Parameters:
    - res4d_path: path to 4D residuals NIfTI file
    - mask_path: path to binary brain mask
    
    Returns:
    - Dictionary with FWHM (mm), DLH, RESELS, and V (voxels)
    """
    # res_img = nib.load(res4d_path)
    # mask_img = nib.load(mask_path)

    # res_data = res_img.get_fdata()
    # mask = mask_img.get_fdata().astype(bool)
    # affine = res_img.affine
    # voxel_dims = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    voxel_dims = hdr['pixdim'][1:4]
    fx,fy,fz=hdr['dim'][1:4]
    fxyz=fx*fy*fz
    ft=res4d.shape[3]


    # compute derivatives
    dx = np.diff(res4d, axis=0)
    dy = np.diff(res4d, axis=1)
    dz = np.diff(res4d, axis=2)
    dx=np.reshape(dx,((fx-1)*fy*fz,ft))
    dy=np.reshape(dy,(fx*(fy-1)*fz,ft))
    dz=np.reshape(dz,(fx*fy*(fz-1),ft))

    # restrict to mask (reduce all by 1 voxel along that axis)
    msk_dx = mask[:-1, :, :].reshape(dx.shape[0])
    msk_dy = mask[:, :-1, :].reshape(dy.shape[0])
    msk_dz = mask[:, :, :-1].reshape(dz.shape[0])
    
    dx=dx[msk_dx==1,:]
    dy=dy[msk_dy==1,:]
    dz=dz[msk_dz==1,:]
    
    del msk_dx,msk_dy,msk_dz

    # compute variance of residuals and gradients
    res4d=np.reshape(res4d,(fxyz,ft))
    mask=np.reshape(mask,(fxyz))
    res4d=res4d[mask==1,:]
    # var = res4d[mask].reshape(-1)  # residual variance
    ssq = np.var(res4d)

    ddx = np.var(dx)
    ddy = np.var(dy)
    ddz = np.var(dz)

    # compute FWHM in mm
    fwhm_x = np.sqrt(4 * np.log(2) * ssq / ddx) * voxel_dims[0]
    fwhm_y = np.sqrt(4 * np.log(2) * ssq / ddy) * voxel_dims[1]
    fwhm_z = np.sqrt(4 * np.log(2) * ssq / ddz) * voxel_dims[2]

    fwhm = np.array([fwhm_x, fwhm_y, fwhm_z])

    # Estimate DLH (Determinant of Local Hessian)
    dlh = np.prod(1 / fwhm)

    # Compute volume (number of voxels in mask)
    V = mask.sum()
    
    # Compute RESELS
    resels = V * dlh

    return {
        "FWHM": fwhm,
        "DLH": dlh,
        "RESELS": resels,
        "V": V
    }

zmap=np.zeros(nv,dtype=np.float32)
zmap[mask==1]=tval1[:,10]
zmap=np.reshape(zmap,(fx,fy,fz))
mask=np.reshape(mask,(fx,fy,fz))

ptfce(zmap, mask=mask, V=smo["V"], Rd=1 / smo["DLH"])
