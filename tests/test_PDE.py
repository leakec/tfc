import jax.numpy as np
from jax import vmap, jacfwd, jit

from tfc import mtfc as TFC
from tfc.utils import NLLS, egrad

def test_PDE():
    ## TFC Parameters
    maxIter = 10
    tol = 1e-13

    # Constants and switches:
    n = 20
    m = 20
    x0 = np.array([0.,0.])
    xf = np.array([1.,1.])

    # Real analytical solution:
    real = lambda x,y: y**2*np.sin(np.pi*x)

    # Create the TFC Class:
    N = np.array([n,n])
    nC = np.array([2,2])
    tfc = TFC(N,nC,m,x0=x0,xf=xf,dim=2,basis='CP')
    x = tfc.x

    Zero = np.zeros_like(x[0])
    One = np.ones_like(x[0])

    # Get the basis functions
    H = tfc.H
    Hy = tfc.Hy

    z1 = lambda xi,*x: np.dot(H(*x),xi)-(1.-x[0])*np.dot(H(*(Zero,x[1])),xi)-x[0]*np.dot(H(*(One,x[1])),xi)
    z = lambda xi,*x: z1(xi,*x)-z1(xi,x[0],Zero)+x[1]*(2.*np.sin(np.pi*x[0])-egrad(z1,2)(xi,x[0],One))

    # Create the residual
    zxx = egrad(egrad(z,1),1)
    zyy = egrad(egrad(z,2),2)
    zy = egrad(z,2)

    r = lambda xi,*x: zxx(xi,*x)+zyy(xi,*x)+z(xi,*x)*zy(xi,*x)-np.sin(np.pi*x[0])*(2.-np.pi**2*x[1]**2+2.*x[1]**3*np.sin(np.pi*x[0]))
    xi = np.zeros(H(*x).shape[1])

    xi,it = NLLS(xi,r,*x,constant_arg_nums=[1,2])

    zr = real(x[0],x[1])
    ze = z(xi,*x)
    err = zr-ze
    maxErr = np.max(np.abs(err))
    assert(maxErr < 1e-10)
