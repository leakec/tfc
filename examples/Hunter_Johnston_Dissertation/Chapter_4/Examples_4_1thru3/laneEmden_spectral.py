# This is a function that solves the general Lane-Emden equation using spectral method 
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
def laneEmden_spectral(N, m, type, xspan, basis, iterMax, tol):
    ## Unpack Paramters: *********************************************************
    x0 = xspan[0]
    xf = xspan[1]

    ## Initial Conditions: *******************************************************
    y0  = 1.
    y0p = 0.
    nC  = 0 # number of constraints

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

    tfc = utfc(N,nC,int(m),basis = basis,x0=x0,xf=xf)
    x = tfc.x
    H = tfc.H

    ## DEFINE THE ASSUMED SOLUTION: *************************************
    y = lambda x,xi: np.dot(H(x),xi)
    yp = egrad(y)
    ypp = egrad(yp)

    ## DEFINE LOSS AND JACOB ********************************************
    L0  = lambda x,xi: y(x,xi)[0] - y0
    Ld0 = lambda x,xi: yp(x,xi)[0] - y0p
    Lde = lambda x,xi: x*ypp(x,xi) + 2.*yp(x,xi) + x*y(x,xi)**type

    L = lambda xi: np.hstack((L0(x,xi),Ld0(x,xi),Lde(x,xi)))
    # J = jit(lambda x,xi: jacfwd(L,1)(x,xi))

    ## SOLVE THE SYSTEM *************************************************
    xi   = np.zeros(H(x).shape[1])

    xi,_,time = NLLS(xi,L,timer=True,maxIter = maxIter)

    ## COMPUTE ERROR AND RESIDUAL ***************************************
    err = np.linalg.norm(y(x,xi) - ytrue(x))
    res = np.linalg.norm(L(xi))

    return err, res, time
