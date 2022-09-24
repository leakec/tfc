import jax.numpy as np
from jax import vmap, jacfwd, jit, jacrev

from tfc import utfc as TFC
from tfc.utils import egrad,NLLS

def test_ODE_Cpp():
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
    m = 30
    nC = 2
    tfc = TFC(N,nC,m,x0=ti,xf=tf,basis='LeP')
    t = tfc.x

    # Get the Chebyshev polynomials
    H = tfc.H
    dH = tfc.dH

    Zero = np.zeros_like(t)
    End = tf*np.ones_like(t)

    H0 = H(Zero)
    Hd0 = dH(Zero)
    Hf = H(End)

    # Create the constraint expression and its derivatives
    beta0 = lambda t: (t-tf)/(ti-tf)
    beta1 = lambda t: (ti-t)/(ti-tf)
    y = lambda t,xi: np.dot(H(t),xi)+beta0(t)*(yi-np.dot(H0,xi))+beta1(t)*(yf-np.dot(Hf,xi))

    yd = egrad(y)
    ydd = egrad(yd)

    # Create the residual and jacobians
    r = lambda xi,t: ydd(t,xi)+f(t)*y(t,xi)*yd(t,xi)-f2(t)
    xi = np.zeros(H(t).shape[1])

    xi,it = NLLS(xi,r,t,constant_arg_nums=[1])

    assert(np.max(np.abs(r(xi,t))) < 1e-10)

def test_ODE_Python():
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
    m = 30
    nC = 2
    tfc = TFC(N,nC,m,x0=ti,xf=tf,basis='LeP',backend="Python")
    t = tfc.x

    # Get the Chebyshev polynomials
    H = tfc.H
    dH = tfc.dH

    Zero = np.zeros_like(t)
    End = tf*np.ones_like(t)

    H0 = H(Zero)
    Hd0 = dH(Zero)
    Hf = H(End)

    # Create the constraint expression and its derivatives
    beta0 = lambda t: (t-tf)/(ti-tf)
    beta1 = lambda t: (ti-t)/(ti-tf)
    y = lambda t,xi: np.dot(H(t),xi)+beta0(t)*(yi-np.dot(H0,xi))+beta1(t)*(yf-np.dot(Hf,xi))

    yd = egrad(y)
    ydd = egrad(yd)

    # Create the residual and jacobians
    r = lambda xi,t: ydd(t,xi)+f(t)*y(t,xi)*yd(t,xi)-f2(t)
    xi = np.zeros(H(t).shape[1])

    xi,it = NLLS(xi,r,t,constant_arg_nums=[1])

    assert(np.max(np.abs(r(xi,t))) < 1e-10)
