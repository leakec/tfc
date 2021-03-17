# This is a function that solves the general Lane-Emden equation using TFC
# Hunter Johnston - Texas A&M University
# Updated: 10 Mar 2021
##################################################################
# Differential Equation
#   y'' + 2/x y' + y^k = 0
#
#   subject to: y(0) = 1, y'(0) = 0
##################################################################
from tfc import utfc
from tfc.utils import egrad, NLLS
import jax.numpy as np
from jax import jit

import numpy as onp
import tqdm
##################################################################
def laneEmden_tfc(N, m, type, xspan, basis, iterMax, tol):
    ## Unpack Paramters: *********************************************************
    x0 = xspan[0]
    xf = xspan[1]

    ## Initial Conditions: *******************************************************
    y0  = 1.
    y0p = 0.
    nC  = 2 # number of constraints

    ## Determine call tfc class needs to be 1 for ELMs
    if basis == 'CP' or basis == 'LeP':
        c = 2./ (xf - x0)
    elif basis == 'FS':
        c = 2. * np.pi / (xf - x0)
    else:
        c = 1./ (xf - x0)

    ## Compute true solution
    if type == 0:
        maxIter = 1
        def ytrue(x):
            val = onp.zeros_like(x)
            val[0] = 1.
            val[1:] = 1. - 1./6. * x[1:]**2
            return val
    elif type == 1:
        maxIter = 1
        def ytrue(x):
            val = onp.zeros_like(x)
            val[0] = 1.
            val[1:] = np.sin(x[1:]) / x[1:]
            return val

    elif type == 5:
        maxIter = iterMax
        def ytrue(x):
            val = onp.zeros_like(x)
            val[0] = 1.
            val[1:] = (1. + x[1:]**2/3)**(-1/2)
            return val
    else:
        def ytrue(x):
            return np.nan * np.ones_like(x)

    err = np.ones_like(m) * np.nan
    res = np.ones_like(m) * np.nan

    ## GET CHEBYSHEV VALUES: *********************************************

    tfc = utfc(N,nC,int(m),basis = basis, x0=x0, xf=xf)
    x = tfc.x

    H = tfc.H
    dH = tfc.dH
    H0 = H(x[0])
    H0p = dH(x[0])

    ## DEFINE THE ASSUMED SOLUTION: *************************************
    phi1 = lambda x: np.ones_like(x)
    phi2 = lambda x: x

    y = lambda x,xi: np.dot(H(x),xi) \
                    + phi1(x)*(y0  - np.dot(H0,xi)) \
                    + phi2(x)*(y0p - np.dot(H0p,xi))
    yp = egrad(y)
    ypp = egrad(yp)

    ## DEFINE LOSS AND JACOB ********************************************
    L = jit(lambda xi: x*ypp(x,xi) + 2.*yp(x,xi) + x*y(x,xi)**type)

    ## SOLVE THE SYSTEM *************************************************

    # Solve the problem
    xi = np.zeros(H(x).shape[1])

    xi,_,time = NLLS(xi,L,timer=True,maxIter = maxIter)

    ## COMPUTE ERROR AND RESIDUAL ***************************************
    err = np.linalg.norm(y(x,xi) - ytrue(x))
    res = np.linalg.norm(L(xi))
    
    return err, res, time
