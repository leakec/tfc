# This is a function that solves the BVP using TFC
# Hunter Johnston - Texas A&M University
# Updated: 15 Mar 2021
#####################################################################
# Differential Equation
#   y'' + yy' = exp(-2x) sin(x) [cos(x) -sin(x)] - 2exp(-x)cos(x)
#
#   subject to: y(0) = 0, y(pi) = 0
#####################################################################
from tfc import utfc
from tfc.utils import egrad, NLLS
import jax.numpy as np
from jax import jit

import numpy as onp
import tqdm
#####################################################################

def BVP_tfc(N, m, basis, iterMax, tol):
    ## Unpack Paramters: *********************************************************
    x0 = 0.
    xf = np.pi

    ## Initial Conditions: *******************************************************
    y0  = 0.
    yf  = 0.
    nC  = 2 # number of constraints

    ## Determine call tfc class needs to be 1 for ELMs
    if basis == 'CP' or basis == 'LeP':
        c = 2./ (xf - x0)
    elif basis == 'FS':
        c = 2. * np.pi / (xf - x0)
    else:
        c = 1./ (xf - x0)

    ## Compute true solution
    ytrue = lambda x: np.exp(-x) * np.sin(x)

    err = onp.ones_like(m) * np.nan
    res = onp.ones_like(m) * np.nan

    ## GET CHEBYSHEV VALUES: *********************************************

    tfc = utfc(N,nC,int(m),basis = basis,x0=x0, xf=xf)
    x = tfc.x

    H = tfc.H
    H0 = H(tfc.x[0])
    Hf = H(tfc.x[-1])

    ## DEFINE THE ASSUMED SOLUTION: *************************************
    phi1 = lambda x: (np.pi - x)/np.pi
    phi2 = lambda x: x/np.pi

    f = lambda x: np.exp(-2.*x) * np.sin(x) * (np.cos(x) - np.sin(x)) - 2.*np.exp(-x)*np.cos(x)

    y = lambda x,xi: np.dot(H(x),xi) + phi1(x)*(y0 - np.dot(H0,xi)) + phi2(x)*(yf - np.dot(Hf,xi))
    yp = egrad(y)
    ypp = egrad(yp)

    ## DEFINE LOSS AND JACOB ********************************************
    L = lambda xi: ypp(x,xi) + y(x,xi)*yp(x,xi) - f(x)

    ## SOLVE THE SYSTEM *************************************************
    xi = onp.zeros(H(x).shape[1])

    xi,_,time = NLLS(xi,L,timer=True,maxIter=iterMax)

    ## COMPUTE ERROR AND RESIDUAL ***************************************
    err = onp.linalg.norm(y(x,xi) - ytrue(x))
    res = onp.linalg.norm(L(xi))

    return err, res, time
