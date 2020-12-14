import tqdm
import numpy as onp
import jax.numpy as np
from jax import jacfwd
import matplotlib.pyplot as plt 

from tfc import mtfc
from tfc.utils import egrad, MakePlot
from tfc.utils.Latex import table

# Constants and switches:
nMC = 100

n = 30 
m = 347 # Change to 17 to get the second histogram

x0 = np.array([0.,0.])
xf = np.array([1.,1.])

# Allocate memory
testErr = onp.zeros(nMC)

# Real analytical solution:
real = lambda x,y: np.exp(-x)*(x+y**3)

# Create the TFC Class:
N = [n,]*2
nC = [-1,]
tfc = mtfc(N,nC,m,dim=2,basis='ELMTanh',x0=x0,xf=xf)
x = tfc.x

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

# Create least-squares function
zXi = np.zeros(H(*x).shape[1])
def LS():
    A = jacfwd(L,0)(zXi,*x)
    B = -L(zXi,*x)
    xi = np.linalg.lstsq(A,B,rcond=None)[0]
    return xi

# Create the test set 
dark = np.meshgrid(np.linspace(x0[0],xf[0],n),np.linspace(x0[1],xf[1],n))
xTest = (dark[0].flatten(),dark[1].flatten())

# Solve the problem for the various n and m values
for k in tqdm.trange(nMC):

    # Solve for xi
    tfc.basisClass.w = np.array(2.*onp.random.rand(*tfc.basisClass.w.shape)-1.)
    tfc.basisClass.b = np.array(2.*onp.random.rand(*tfc.basisClass.b.shape)-1.)
    xi = LS()

    # Calculate the error
    ur = real(*xTest)
    ue = u(xi,*xTest)
    err = ur-ue
    testErr[k] = np.max(np.abs(err))

p1 = MakePlot('Maximum Error','Number of Occurances')
hist,binEdge = np.histogram(np.log10(testErr),bins=20)
p1.ax[0].hist(testErr,bins=10**binEdge,color=(76./256.,0.,153./256.),edgecolor='black',zorder=20)
p1.ax[0].set_xscale('log')
p1.ax[0].xaxis.set_major_locator(plt.LogLocator(base=10,numticks=10))
p1.ax[0].locator_params(axis='both',tight=True)
p1.ax[0].grid(True,which='both')
[line.set_zorder(0) for line in p1.ax[0].lines]
mTicks = p1.ax[0].xaxis.get_minor_ticks()
p1.PartScreen(11,8)
p1.show()
