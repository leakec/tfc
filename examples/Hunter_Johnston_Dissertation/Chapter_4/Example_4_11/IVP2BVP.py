# This is a function that solves the IVP to BVP problem
# Hunter Johnston - Texas A&M University
# Updated: 3 Sep 2020
##################################################################################
# Differential Equation
#
#
#   subject to:
##################################################################################
from tfc import utfc
from tfc.utils import TFCDict, egrad, NLLS

import numpy as onp
import jax.numpy as np
from jax import vmap, jacfwd, jit, lax

import tqdm

from timeit import default_timer as timer
##################################################################################

def IVP2BVP(N, m, gamma, basis, iterMax, tol):
    ## Unpack Paramters: *********************************************************
    x0 = -1.
    xf =  1.

    ## Initial Conditions: *******************************************************
    y0  = -2.
    y0p = -2.
    yf  =  2.

    nC  =  2 # number of constraints

    ## Determine call tfc class needs to be 1 for ELMs
    if basis == 'CP' or basis == 'LeP':
        c = 2./ (xf - x0)
    elif basis == 'FS':
        c = 2. * np.pi / (xf - x0)
    else:
        c = 1./ (xf - x0)

    ## GET CHEBYSHEV VALUES: *********************************************

    tfc = utfc(N,nC,m,basis = basis, x0=-1., xf=1.)
    x = tfc.x

    H = tfc.H
    dH = tfc.dH
    H0 = H(tfc.z[0:1])
    Hf = H(tfc.z[-1:])
    H0p = dH(tfc.z[0:1])

    ## DEFINE THE ASSUMED SOLUTION: *************************************
    phi1 = lambda a: 1./(1. + 4.*gamma - gamma**2) * ( (1. + gamma) - 2.*gamma*a )
    phi2 = lambda a: 1./(1. + 4.*gamma - gamma**2) * ( (1. - gamma)**2 + (1. - gamma**2)*a)
    phi3 = lambda a: 1./(1. + 4.*gamma - gamma**2) * ( -gamma*(gamma-3.) + 2.*gamma*a )

    y = lambda x, xi: np.dot(H(x),xi) + phi1(x)*(y0  - np.dot(H0,xi)) \
                                      + phi2(x)*(y0p - np.dot(H0p,xi)) \
                                      + phi3(x)*(yf  - np.dot(Hf,xi))
    yp = egrad(y,0)
    ypp = egrad(yp,0)


    ## DEFINE LOSS AND JACOB ********************************************
    L = lambda xi: ypp(x,xi) + (np.cos(3.*x**2) -3.*x + 1.)*yp(x,xi) \
                             + (6.*np.sin(4.*x**2) - np.exp(np.cos(3.*x)))*y(x,xi) \
                             - 2. * (1.-np.sin(3.*x))*(3.*x-np.pi)/(4.-x)



    ## SOLVE THE SYSTEM *************************************************
    xi   = onp.zeros(H(x).shape[1])


    xi,_,_ = NLLS(xi,L,timer=True)

    return y(x,xi), L(xi), x
