#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 21:20:45 2025

@author: WauWter
"""
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.ndimage import label
from scipy.special import gamma
from tqdm import tqdm
# nibabel for NIfTI I/O
# scikit-image or scipy.ndimage for cluster components


def aggregate_logpvals(logpvals, d):
    logpvals[np.isinf(logpvals)] = 745
    s = logpvals.sum()
    return 0.5 * (np.sqrt(d * (8 * s + d)) - d)

def Es(h, V, Rd):
    h2 = h * h
    ret = np.log(V) + norm.logsf(h)
    mask = h >= 1.1
    if mask.any():
        h2m = h2[mask]
        ret[mask] -= (
            np.log(Rd) + np.log(h2m - 1) + h2m / 2 + 2 * np.log(2 * np.pi)
        )
    return np.exp(ret)

def dvox(h): return norm.pdf(h)
def pvox(h): return norm.sf(h)

def dclust(h, V, Rd, c, ZestThr=1.3, CONST=1e40):
    def inner(h_val):
        if h_val < ZestThr:
            return 0
        lam = (Es(np.array([h_val]), V, Rd)[0] / gamma(2.5)) ** (-2/3)
        return lam * np.exp(-lam * c**(2/3))
    vfunc = np.vectorize(inner)
    return vfunc(h) * CONST

def dvox_clust(h, V, Rd, c, ZestThr=1.3):
    num = dvox(h) * dclust(h, V, Rd, c, ZestThr)
    denom, _ = quad(lambda x: dvox(x) * dclust(x, V, Rd, c, ZestThr), -np.inf, np.inf, limit=500)
    return num / denom if denom > 0 else dvox(h)

def pvox_clust(V, Rd, c, actH, ZestThr=1.3):
    if actH <= ZestThr:
        return pvox(actH)
    val, _ = quad(lambda x: dvox_clust(x, V, Rd, c, ZestThr), actH, np.inf)
    return val

def ptfce(img, mask, Rd=None, V=None, resels=None, residual=None, dof=None,
          logpmin=0, logpmax=None, ZestThr=1.3, Nh=100, verbose=True):
    # 1. Clean NAs / Infs
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Smoothness estimation if needed (placeholder)
    if Rd is None or V is None:
        # autosmooth = True
        # Call smoothest(...) to get V and Rd
        raise NotImplementedError("Implement smoothest estimation here")

    # 3. Threshold grid setup
    if logpmax is None:
        logpmax = -np.log(norm.sf(img.max()))
    logp_thres = np.linspace(logpmin, logpmax, Nh)
    dh = logp_thres[1] - logp_thres[0]
    p_thres = np.exp(-logp_thres)
    threshs = norm.ppf(1 - p_thres)
    threshs[0] = -np.inf

    # 4. Arrays for cluster sizes and p-values
    shape = img.shape
    CLUST = np.zeros((Nh, *shape), dtype=int)
    PVC = np.ones((Nh, *shape), dtype=float)

    # 5. Loop thresholds & compute component-wise
    for i, h in enumerate(tqdm(threshs, disable=not verbose)):
        bw = img > h
        labeled, n_comp = label(bw, structure=np.ones((3,3,3)))
        sizes = np.bincount(labeled.flat)
        size_map = sizes[labeled]
        CLUST[i] = size_map * mask

        # Compute PVC per voxel cluster size
        unique_sizes = np.unique(size_map)
        for c in unique_sizes:
            if c == 0: continue
            mask_c = (size_map == c)
            PVC[i][mask_c] = pvox_clust(V, Rd, c, h, ZestThr)

    # 6. Aggregate over thresholds
    logp_array = -np.log(PVC)
    pTFCE = np.exp(-aggregate_logpvals(logp_array.reshape((Nh, -1)), dh))
    pTFCE = pTFCE.reshape(shape)

    # 7. Convert to Z, handle under/overflow
    pTFCE = np.clip(pTFCE, np.finfo(float).tiny, 1 - np.finfo(float).eps)
    Z = norm.isf(pTFCE)

    # 8. FWER correction threshold Z via fwe.p2z
    # Implement fwe.p2z() in Python
    # fwer0_05_Z = fwe_p2z(V/resels if resels else None)

    # return {'p': pTFCE, 'Z': Z, 'fwer0.05_Z': fwer0_05_Z}
    return {'p': pTFCE, 'Z': Z}
