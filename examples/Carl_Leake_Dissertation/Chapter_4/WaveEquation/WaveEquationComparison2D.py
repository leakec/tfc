import tqdm

import numpy as onp
import jax.numpy as np
from jax import jacfwd, jit

from tfc import mtfc
from tfc.utils import egrad
from tfc.utils.Latex import table

# Constants and switches:
n = 11
nTest = 15

x0 = np.array([0.,0.,0.])
xf = np.array([1.,1.,1.])

# Real analytical solution:
real = lambda x,y,t: np.sin(np.pi*x)*np.sin(np.pi*y)*np.cos(np.pi*np.sqrt(2.)/8.*t)

def Solve(m,xtfc):

    # TFC Constants:
    if xtfc:
        nC = [-1,]
        basis = 'ELMTanh'
    else:
        nC = [2,2,2]
        basis = 'CP'

    # Create the TFC Class:
    N = [n,n,n]
    myTfc = mtfc(N,nC,m,dim=3,basis=basis,x0=x0,xf=xf)

    # Create the TFC constrained expression
    x = myTfc.x
    H = myTfc.H

    u1 = lambda xi,*x: np.dot(H(*x),xi)-(1.-x[0])*np.dot(H(np.zeros_like(x[0]),x[1],x[2]),xi)-x[0]*np.dot(H(np.ones_like(x[0]),x[1],x[2]),xi)
    u2 = lambda xi,*x: u1(xi,*x)-(1.-x[1])*u1(xi,x[0],np.zeros_like(x[1]),x[2])-x[1]*u1(xi,x[0],np.ones_like(x[1]),x[2])
    du2dt = egrad(u2,3)
    u = lambda xi,*x: u2(xi,*x)+np.sin(np.pi*x[0])*np.sin(np.pi*x[1])-u2(xi,x[0],x[1],np.zeros_like(x[2]))-x[2]*du2dt(xi,x[0],x[1],np.zeros_like(x[2]))

    # Create the residual
    uxx = egrad(egrad(u,1),1)
    uyy = egrad(egrad(u,2),2)
    utt = egrad(egrad(u,3),3)

    r = lambda xi: uxx(xi,*x)+uyy(xi,*x)-64.*utt(xi,*x)
    xi = np.zeros(H(*x).shape[1])

    if xtfc:
        LS = lambda xi: np.linalg.lstsq(jacfwd(r,0)(xi),-r(xi),rcond=None)[0]
    else:
        LS = lambda xi: np.dot(np.linalg.pinv(jacfwd(r,0)(xi)),-r(xi))

    xi = LS(xi)

    # Calculate the test set error
    dark = np.meshgrid(np.linspace(x0[0],xf[0],nTest),
                       np.linspace(x0[1],xf[1],nTest),
                       np.linspace(x0[2],xf[2],nTest))
    xTest = tuple([k.flatten() for k in dark])
    err = np.abs(real(*xTest)-u(xi,*xTest))
    return np.max(err),np.mean(err),myTfc.basisClass.numBasisFunc

m = [3,6,9,12,15,18]
mLen = len(m)
basisFunc = onp.zeros(mLen)
tfcMaxError = onp.zeros(mLen)
tfcMeanError = onp.zeros(mLen)
xtfcMaxError = onp.zeros(mLen)
xtfcMeanError = onp.zeros(mLen)
for k in tqdm.trange(mLen):
    tfcMaxError[k],tfcMeanError[k],basisFunc[k] = Solve(m[k],False)
    xtfcMaxError[k],xtfcMeanError[k],dark = Solve(int(basisFunc[k]-1),True)
    assert(dark == basisFunc[k])

data = np.vstack([basisFunc,tfcMaxError,tfcMeanError,xtfcMaxError,xtfcMeanError]).T
tab = table.SimpleTable(data,form='%.2e')
print(tab)
