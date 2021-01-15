import numpy as onp
import jax.numpy as np
from jax import jacfwd, jit
from matplotlib import cm

from tfc import mtfc
from tfc.utils import LS, egrad, MakePlot

# Constants and switches:
n = 30
x0 = np.array([0.,0.])
xf = np.array([1.,1.])

xtfc = False

# Real analytical solution:
real = lambda x,t: np.sin(np.pi*x)*np.cos(np.pi*t)

# TFC Constants:
if xtfc:
    m = 200
    nC = -1
    basis = 'ELMTanh'
else:
    m = 20
    nC = [2,2]
    basis = 'LeP'

# Create the TFC Class:
N = [n,n]
myTfc = mtfc(N,nC,m,dim=2,basis=basis,x0=x0,xf=xf)

# Create the constrained expression:
H = myTfc.H
x = myTfc.x

u1 = lambda xi,*x: np.dot(H(*x),xi)-(1.-x[0])*np.dot(H(np.zeros_like(x[0]),x[1]),xi)-x[0]*np.dot(H(np.ones_like(x[0]),x[1]),xi)
u1t = egrad(u1,2)
u = lambda xi,*x: u1(xi,*x)+np.sin(np.pi*x[0])-u1(xi,x[0],np.zeros_like(x[1]))-x[1]*u1t(xi,x[0],np.zeros_like(x[1]))

# Create the residual
uxx = egrad(egrad(u,1),1)
utt = egrad(egrad(u,2),2)

L = lambda xi: uxx(xi,*x)-utt(xi,*x)

# Solve the problem
xi = np.zeros(H(*x).shape[1])

if xtfc:
    xi,time = LS(xi,L,method='lstsq',timer=True)
else:
    xi,time = LS(xi,L,timer=True)

# Calculate the test set error
nTest = 100
dark = np.meshgrid(np.linspace(x0[0],xf[0],nTest),np.linspace(x0[1],xf[1],nTest))
xTest = (dark[0].flatten(),dark[1].flatten())
err = np.abs(real(*xTest)-u(xi,*xTest))

# Print out solution statistics
print("Time: "+str(time))
print("Max error test: "+str(np.max(err)))
print("Mean error test: "+str(np.mean(err)))

# Create plots
p = MakePlot(r'$x$',r'$t$',zlabs=r'$u(x,t)$')
p.ax[0].plot_surface(x[0].reshape((n,n)),x[1].reshape((n,n)),real(*x).reshape((n,n)),
                     cmap=cm.gist_rainbow,antialiased=False,rcount=n,ccount=n)

p.ax[0].tick_params(axis='z', which='major', pad=10)
p.ax[0].xaxis.labelpad = 20
p.ax[0].yaxis.labelpad = 20
p.ax[0].zaxis.labelpad = 20
p.ax[0].view_init(azim=-25,elev=25)

p.PartScreen(8,7)
p.show()

p1 = MakePlot('x','y',zlabs='error')
p1.ax[0].plot_surface(*dark,err.reshape((nTest,nTest)),cmap=cm.gist_rainbow)
p1.FullScreen()
p1.show()
