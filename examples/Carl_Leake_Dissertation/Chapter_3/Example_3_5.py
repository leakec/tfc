# Import python packages
from tqdm import tqdm
import numpy as onp
import jax.numpy as np
from jax import jacfwd
from matplotlib import cm

# Import TFC classes
from tfc import mtfc
from tfc.utils import egrad
from tfc.utils.Latex import table

# Constants and switches:
nVec = [5,10,15,20,25,30]
mVec = [17,62,132,227,347]

x0 = np.array([0.,0.])
xf = np.array([1.,1.])

# Allocate memory
testErr = onp.zeros((len(nVec),len(mVec)))

# Real analytical solution:
real = lambda x,y: np.exp(-x)*(x+y**3)

# Solve the problem for the various n and m values
for j,n in enumerate(tqdm(nVec)):
    for k,m in enumerate(mVec):

        # Create the TFC Class:
        N = [n,]*2
        nC = -1
        tfc = mtfc(N,nC,m,dim=2,basis='ELMTanh',x0=x0,xf=xf)
        x = tfc.x

        if tfc.basisClass.numBasisFunc > n**2:
            testErr[j,k] = np.nan
            continue

        # Get the basis functions
        H = tfc.H

        # Create the constrained expression
        u1 = lambda xi,*x: np.dot(H(*x),xi)\
                           +(1.-x[0])*(x[1]**3-np.dot(H(np.zeros_like(x[0]),x[1]),xi))\
                           +x[0]*((1.+x[1]**3)*np.exp(-1.)-np.dot(H(np.ones_like(x[0]),x[1]),xi))
        u = lambda xi,*x: u1(xi,*x)\
                          +(1.-x[1])*(x[0]*np.exp(-x[0])-u1(xi,x[0],np.zeros_like(x[1])))\
                          +x[1]*(np.exp(-x[0])*(x[0]+1.)-u1(xi,x[0],np.ones_like(x[1])))

        # Create the residual
        laplace = lambda xi,*x: egrad(egrad(u,1),1)(xi,*x)+egrad(egrad(u,2),2)(xi,*x)
        L = lambda xi,*x: laplace(xi,*x)-np.exp(-x[0])*(x[0]-2.+x[1]**3+6.*x[1])

        # Calculate the xi values
        zXi = np.zeros(H(*x).shape[1])
        A = jacfwd(L,0)(zXi,*x)
        B = -L(zXi,*x)
        xi = np.linalg.lstsq(A,B,rcond=None)[0]

        # Calculate the error
        dark = np.meshgrid(np.linspace(x0[0],xf[0],n),np.linspace(x0[1],xf[1],n))
        x = (dark[0].flatten(),dark[1].flatten())

        ur = real(*x)
        ue = u(xi,*x)
        err = ur-ue
        testErr[j,k] = np.max(np.abs(err))

# Print results as a table
tab = table.SimpleTable(testErr)
print(tab)
f = open("XtfcData.txt","w")
f.write(tab)
f.close()
