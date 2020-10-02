import os,sys
sourcePath = os.path.join("..","src","build","bin")
sys.path.append(sourcePath)

import jax.numpy as np
from jax import vmap, jacfwd, jit, jacrev

from TFC import TFC
from TFCUtils import NLLS

def test_ODE():
    # This script will solve the non-linear differential equation 
    # of the form: y''+f(t)*y*y' = f2(t)

    # Constants used in the differential equation:
    f = lambda t: np.ones(t.shape)
    f2 = lambda t: np.exp(-2.*t)*np.sin(t)*(np.cos(t)-np.sin(t))-2.*np.exp(-t)*np.cos(t)

    ti = 0.
    tf = np.pi
    yi = 0.
    yf = 0.

    # Real analytical solution:
    real = lambda t: np.exp(-t)*np.sin(t)

    # Create the ToC Class:
    N = 100
    c = 2./(tf-ti)
    m = 30
    nC = 2
    tfc = TFC(N,nC,m,basis='LeP',c=c)

    # Get the Chebyshev polynomials
    H = tfc.H
    dH = tfc.dH
    H0 = tfc.H(tfc.z[0],useVal=True)
    Hd0 = tfc.dH(tfc.z[0],useVal=True)
    Hf = tfc.H(tfc.z[N-1],useVal=True)

    # Create the constraint expression and its derivatives
    t = tfc.x
    beta0 = lambda t: (t-tf)/(ti-tf)
    beta1 = lambda t: (ti-t)/(ti-tf)
    y = lambda t,xi: np.dot(H(t),xi)+beta0(t)*(yi-np.dot(H0,xi))+beta1(t)*(yf-np.dot(Hf,xi))

    yd = tfc.egrad(y)
    ydd = tfc.egrad(yd)

    # Create the residual and jacobians
    r = jit(lambda xi: ydd(t,xi)+f(t)*y(t,xi)*yd(t,xi)-f2(t))
    xi = np.zeros(H(t).shape[1])

    xi,it = NLLS(xi,r)

    assert(np.max(np.abs(r(xi))) < 1e-10)
