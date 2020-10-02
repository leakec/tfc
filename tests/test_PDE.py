import os,sys
sourcePath = os.path.join("..","src","build","bin")
sys.path.append(sourcePath)

import jax.numpy as np
from jax import vmap, jacfwd, jit

from nTFC import TFC
from TFCUtils import NLLS

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
    c = 2./(xf-x0)
    tfc = TFC(N,nC,m,dim=2,basis='CP',c=c)
    x = tfc.x

    # Get the basis functions
    H = tfc.H
    Hy = tfc.Hy

    H00 = H(*(tfc.RepMat(tfc.z[0,0]),tfc.RepMat(tfc.z[1,0])),useVal=np.array([1,1],dtype=np.int32))
    H10 = H(*(tfc.RepMat(tfc.z[0,-1]),tfc.RepMat(tfc.z[1,0])),useVal=np.array([1,1],dtype=np.int32))
    Hy01 = Hy(*(tfc.RepMat(tfc.z[0,0]),tfc.RepMat(tfc.z[1,-1])),useVal=np.array([1,1],dtype=np.int32))
    Hy11 = Hy(*(tfc.RepMat(tfc.z[0,-1]),tfc.RepMat(tfc.z[1,-1])),useVal=np.array([1,1],dtype=np.int32))

    def Hx0(*x):
        return H(*(x[0],tfc.RepMat(tfc.z[1,0])),useVal=np.array([0,1],dtype=np.int32))
    def Hyx1(*x):
        return Hy(*(x[0],tfc.RepMat(tfc.z[1,-1])),useVal=np.array([0,1],dtype=np.int32))
    def H0y(*x):
        return H(*(tfc.RepMat(tfc.z[0,0]),x[1]),useVal=np.array([1,0],dtype=np.int32))
    def H1y(*x):
        return H(*(tfc.RepMat(tfc.z[0,-1]),x[1]),useVal=np.array([1,0],dtype=np.int32))

    # Create the TFC constrained expression
    z = lambda xi,*x: np.dot(H(*x),xi)+(1.-x[0])*np.dot(H00,xi)-(1.-x[0])*np.dot(H0y(*x),xi)+x[0]*np.dot(H10,xi)-x[0]*np.dot(H1y(*x),xi)-np.dot(Hx0(*x),xi)+x[1]*(2.*np.sin(np.pi*x[0])+(1.-x[0])*np.dot(Hy01,xi)+x[0]*np.dot(Hy11,xi)-np.dot(Hyx1(*x),xi))

    # Create the residual
    zxx = tfc.egrad(tfc.egrad(z,1),1)
    zyy = tfc.egrad(tfc.egrad(z,2),2)
    zy = tfc.egrad(z,2)

    r = lambda xi: zxx(xi,*x)+zyy(xi,*x)+z(xi,*x)*zy(xi,*x)-np.sin(np.pi*x[0])*(2.-np.pi**2*x[1]**2+2.*x[1]**3*np.sin(np.pi*x[0]))
    xi = np.zeros(H(*x).shape[1])

    xi,it = NLLS(xi,r)

    zr = real(x[0],x[1])
    ze = z(xi,*x)
    err = zr-ze
    maxErr = np.max(np.abs(err))
    assert(maxErr < 1e-10)
