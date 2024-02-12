#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 17:05:32 2023

@author: WauWter (who copied this from NiPy, just so you know)



This module provides definitions of various hemodynamic response
functions (hrf).

In particular, it provides Gary Glover's canonical HRF, AFNI's default
HRF, and a spectral HRF.

The Glover HRF is based on:

@article{glover1999deconvolution,
  title={{Deconvolution of impulse response in event-related BOLD fMRI}},
  author={Glover, G.H.},
  journal={NeuroImage},
  volume={9},
  number={4},
  pages={416--429},
  year={1999},
  publisher={Orlando, FL: Academic Press, c1992-}
}

This parametrization is from fmristat:

http://www.math.mcgill.ca/keith/fmristat/

fmristat models the HRF as the difference of two gamma functions, ``g1``
and ``g2``, each defined by the timing of the gamma function peaks
(``pk1, pk2``) and the FWHMs (``width1, width2``):

   raw_hrf = g1(pk1, width1) - a2 * g2(pk2, width2)

where ``a2`` is the scale factor for the ``g2`` gamma function.  The
actual hrf is the raw hrf set to have an integral of 1. 

fmristat used ``pk1, width1, pk2, width2, a2 = (5.4 5.2 10.8 7.35
0.35)``.  These are parameters to match Glover's 1 second duration
auditory stimulus curves.  Glover wrote these as:

   y(t) = c1 * t**n1 * exp(t/t1) - a2 * c2 * t**n2 * exp(t/t2)

with ``n1, t1, n2, t2, a2 = (6.0, 0.9, 12, 0.9, 0.35)``,  and ``c1, c2`` being
``1/max(t**n1 * exp(t/t1)), 1/max(t**n2 * exp(t/t2)``.  The difference between
Glover's expression and ours is because we (and fmristat) use the peak location
and width to characterize the function rather than ``n1, t1``.  The values we
use are equivalent.  Specifically, in our formulation:

>>> n1, t1, c1 = gamma_params(5.4, 5.2)
>>> np.allclose((n1-1, t1), (6.0, 0.9), rtol=0.02)
True
>>> n2, t2, c2 = gamma_params(10.8, 7.35)
>>> np.allclose((n2-1, t2), (12.0, 0.9), rtol=0.02)
True
"""

from functools import partial

import numpy as np
import sympy

import scipy.stats as sps

from sympy.utilities.lambdify import implemented_function, lambdify

#from .utils import lambdify_t, T

class Term(sympy.Symbol):
    """A sympy.Symbol type to represent a term an a regression model

    Terms can be added to other sympy expressions with the single
    convention that a term plus itself returns itself.

    It is meant to emulate something on the right hand side of a formula
    in R. In particular, its name can be the name of a field in a
    recarray used to create a design matrix.

    >>> t = Term('x')
    >>> xval = np.array([(3,),(4,),(5,)], np.dtype([('x', np.float)]))
    >>> f = t.formula
    >>> d = f.design(xval)
    >>> print(d.dtype.descr)
    [('x', '<f8')]
    >>> f.design(xval, return_float=True)
    array([ 3.,  4.,  5.])
    """
    # This flag is defined to avoid using isinstance in getterms
    # and getparams.
    _term_flag = True

    def _getformula(self):
        return Formula([self])
    formula = property(_getformula,
                       doc="Return a Formula with only terms=[self].")

    def __add__(self, other):
        if self == other:
            return self
        return sympy.Symbol.__add__(self, other)


# time symbol
T = Term('t')


class Formula(object):
    """ A Formula is a model for a mean in a regression model.

    It is often given by a sequence of sympy expressions, with the mean
    model being the sum of each term multiplied by a linear regression
    coefficient.

    The expressions may depend on additional Symbol instances, giving a
    non-linear regression model.
    """
    # This flag is defined for test isformula(obj) instead of isinstance
    _formula_flag = True

    def __init__(self, seq, char = 'b'):
        """
        Parameters
        ----------
        seq : sequence of ``sympy.Basic``
        char : str, optional
            character for regression coefficient
        """
        self._terms = np.asarray(seq)
        self._counter = 0
        self.char = char

def lambdify_t(expr):
    ''' Return sympy function of t `expr` lambdified as function of t

    Parameters
    ----------
    expr : sympy expr

    Returns
    -------
    func : callable
       Numerical implementation of function
    '''
    return lambdify(T, expr, "numpy")

def gamma_params(peak_location, peak_fwhm):
    """ Parameters for gamma density given peak and width

    TODO: where does the coef come from again.... check fmristat code

    From a peak location and peak FWHM, determine the parameters (shape,
    scale) of a Gamma density:

    f(x) = coef * x**(shape-1) * exp(-x/scale)

    The coefficient returned ensures that the f has integral 1 over
    [0,np.inf]

    Parameters
    ----------
    peak_location : float
       Location of the peak of the Gamma density
    peak_fwhm : float
       FWHM at the peak

    Returns
    -------
    shape : float
       Shape parameter in the Gamma density
    scale : float
       Scale parameter in the Gamma density
    coef : float
       Coefficient needed to ensure the density has integral 1.
    """
    shape_m1 = np.power(peak_location / peak_fwhm, 2) * 8 * np.log(2.0)
    scale = np.power(peak_fwhm, 2) / peak_location / 8 / np.log(2.0)
    coef = peak_location**(-shape_m1) * np.exp(peak_location / scale)
    return shape_m1 + 1, scale, coef


def gamma_expr(peak_location, peak_fwhm):
    shape, scale, coef = gamma_params(peak_location, peak_fwhm)
    return (
        coef
        * sympy.Piecewise((T + 1e-14, T >= 0), (0, True))**(shape-1)
        * sympy.exp(-(T+1.0e-14)/scale)
        )


def _get_sym_int(f, dt=0.02, t=50):
    # numerical integral of symbolic function
    return _get_num_int(lambdify_t(f), dt, t)


def _get_num_int(lf, dt=0.02, t=50):
    # numerical integral of numerical function
    tt = np.arange(dt,t+dt,dt)
    return lf(tt).sum() * dt


# Glover HRF
_gexpr = gamma_expr(5.4, 5.2) - 0.35 * gamma_expr(10.8, 7.35)
_gexpr = _gexpr / _get_sym_int(_gexpr)
# The numerical function (pass times to get values)
glovert = lambdify_t(_gexpr)
# The symbolic function
glover = implemented_function('glover', glovert)

# Derivative of Glover HRF
_dgexpr = _gexpr.diff(T)
_dpos = sympy.Derivative((T >= 0), T)
_dgexpr = _dgexpr.subs(_dpos, 0)
_dgexpr = _dgexpr / _get_sym_int(sympy.Abs(_dgexpr))
# Numerical function
dglovert = lambdify_t(_dgexpr)
# Symbolic function
dglover = implemented_function('dglover', dglovert)

del(_gexpr); del(_dpos); del(_dgexpr)

# AFNI's HRF
_aexpr = sympy.Piecewise((T, T >= 0), (0, True))**8.6 * sympy.exp(-T/0.547)
_aexpr = _aexpr / _get_sym_int(_aexpr)
# Numerical function
afnit = lambdify_t(_aexpr)
# Symbolic function
afni = implemented_function('afni', afnit)

del(_aexpr)

# SPMs HRF
def spm_hrf_compat(t,
                   peak_delay=6,
                   under_delay=16,
                   peak_disp=1,
                   under_disp=1,
                   p_u_ratio = 6,
                   normalize=True,
                  ):
    """ SPM HRF function from sum of two gamma PDFs

    This function is designed to be partially compatible with SPMs `spm_hrf.m`
    function.

    The SPN HRF is a *peak* gamma PDF (with location `peak_delay` and dispersion
    `peak_disp`), minus an *undershoot* gamma PDF (with location `under_delay`
    and dispersion `under_disp`, and divided by the `p_u_ratio`).

    Parameters
    ----------
    t : array-like
        vector of times at which to sample HRF.
    peak_delay : float, optional
        delay of peak.
    under_delay : float, optional
        delay of undershoot.
    peak_disp : float, optional
        width (dispersion) of peak.
    under_disp : float, optional
        width (dispersion) of undershoot.
    p_u_ratio : float, optional
        peak to undershoot ratio.  Undershoot divided by this value before
        subtracting from peak.
    normalize : {True, False}, optional
        If True, divide HRF values by their sum before returning.  SPM does this
        by default.

    Returns
    -------
    hrf : array
        vector length ``len(t)`` of samples from HRF at times `t`.

    Notes
    -----
    See ``spm_hrf.m`` in the SPM distribution.
    """
    if len([v for v in [peak_delay, peak_disp, under_delay, under_disp]
            if v <= 0]):
        raise ValueError("delays and dispersions must be > 0")
    # gamma.pdf only defined for t > 0
    hrf = np.zeros(t.shape, dtype=np.float32)
    pos_t = t[t > 0]
    peak = sps.gamma.pdf(pos_t,
                         peak_delay / peak_disp,
                         loc=0,
                         scale = peak_disp)
    undershoot = sps.gamma.pdf(pos_t,
                               under_delay / under_disp,
                               loc=0,
                               scale = under_disp)
    hrf[t > 0] = peak - undershoot / p_u_ratio
    if not normalize:
        return hrf
    return hrf / np.sum(hrf)


_spm_can_int = _get_num_int(partial(spm_hrf_compat, normalize=False))


def spmt(t):
    """ SPM canonical HRF, HRF values for time values `t`

    This is the canonical HRF function as used in SPM
    """
    return spm_hrf_compat(t, normalize=False) / _spm_can_int


def dspmt(t):
    """ SPM canonical HRF derivative, HRF derivative values for time values `t`

    This is the canonical HRF derivative function as used in SPM.

    It is the numerical difference of the HRF sampled at time `t` minus the
    values sampled at time `t` -1
    """
    t = np.asarray(t)
    return spmt(t) - spmt(t - 1)


_spm_dd_func = partial(spm_hrf_compat, normalize=False, peak_disp=1.01)
_spm_dd_func_int = _get_num_int(_spm_dd_func)

def ddspmt(t):
    """ SPM canonical HRF dispersion derivative, values for time values `t`

    This is the canonical HRF dispersion derivative function as used in SPM.

    It is the numerical difference between the HRF sampled at time `t`, and
    values at `t` for another HRF shape with a small change in the peak
    dispersion parameter (``peak_disp`` in func:`spm_hrf_compat`).
    """
    return (spmt(t) - _spm_dd_func(t) / _spm_dd_func_int) / 0.01


spm = implemented_function('spm', spmt)
dspm = implemented_function('dspm', dspmt)
ddspm = implemented_function('ddspm', ddspmt)
