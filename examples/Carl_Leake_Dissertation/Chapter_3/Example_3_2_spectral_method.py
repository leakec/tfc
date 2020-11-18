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
mVec = [5,10,15,20,25]

x0 = np.array([0.,0.])
xf = np.array([1.,1.])

testErr = onp.zeros((len(nVec),len(mVec)))

# Real analytical solution:
real = lambda x,y: np.exp(-x)*(x+y**3)

# Solve the problem for the various n and m values
for j,n in enumerate(tqdm(nVec)):
    for k,m in enumerate(mVec):

        # Create the TFC Class:
        N = [n,]*2
        nC = [-1,]*2
        tfc = mtfc(N,nC,m,dim=2,basis='CP',x0=x0,xf=xf)
        x = tfc.x

        if tfc.basisClass.numBasisFunc > n**2:
            testErr[j,k] = np.nan
            continue

        # Get the boundary data points 
        x0ind = np.where(x[0]==0.)[0]
        xfind = np.where(x[0]==1.)[0]
        y0ind = np.where(x[1]==0.)[0]
        yfind = np.where(x[1]==1.)[0]

        # Get the basis functions
        H = tfc.H

        # Create the spectral solution form
        u = lambda xi,*x: np.dot(H(*x),xi)

        # Create the residual
        laplace = lambda xi,*x: egrad(egrad(u,1),1)(xi,*x)+egrad(egrad(u,2),2)(xi,*x)
        L = lambda xi,*x: laplace(xi,*x)-np.exp(-x[0])*(x[0]-2.+x[1]**3+6.*x[1])

        # Calculate the A and B matrices
        zXi = np.zeros((tfc.basisClass.numBasisFunc))
        A = np.vstack([jacfwd(L,0)(zXi,*x),
                       H(x[0][x0ind],x[1][x0ind]),
                       H(x[0][xfind],x[1][xfind]),
                       H(x[0][y0ind],x[1][y0ind]),
                       H(x[0][yfind],x[1][yfind])])
        B = np.hstack([L(zXi,*x),
                       x[1][x0ind]**3,
                       (1.+x[1][xfind]**3)*np.exp(-1.),
                       x[0][y0ind]*np.exp(-x[0][y0ind]),
                       (x[0][yfind]+1.)*np.exp(-x[0][yfind])])

        # Calculate the xi values
        xi = np.dot(np.linalg.pinv(A),B)

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
f = open("SpectralData.txt","w")
f.write(tab)
f.close()
