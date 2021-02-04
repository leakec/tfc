import numpy as onp
import jax.numpy as np
from jax import jacfwd, jit

from tfc import mtfc
from tfc.utils import LS, egrad

# Constants and switches:
n = 11
nTest = 15
x0 = np.array([0.,0.,0.])
xf = np.array([1.,1.,1.])

xtfc = True
usePlotly = True

# Real analytical solution:
real = lambda x,y,t: np.sin(np.pi*x)*np.sin(np.pi*y)*np.cos(np.pi*np.sqrt(2.)/8.*t)

# TFC Constants:
if xtfc:
    nC = -1
    basis = 'ELMTanh'
    m = 650
else:
    m = 18
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

L = lambda xi: uxx(xi,*x)+uyy(xi,*x)-64.*utt(xi,*x)

# Solve the problem
xi = np.zeros(H(*x).shape[1])

if xtfc:
    xi,time = LS(xi,L,method='lstsq',timer=True)
else:
    xi,time = LS(xi,L,timer=True)

# Calculate the test set error
dark = np.meshgrid(np.linspace(x0[0],xf[0],nTest),
                   np.linspace(x0[1],xf[1],nTest),
                   np.linspace(x0[2],xf[2],nTest))
xTest = tuple([k.flatten() for k in dark])
err = np.abs(real(*xTest)-u(xi,*xTest))

# Print out solution statistics
print("Time: "+str(time))
print("Max error test: "+str(np.max(err)))
print("Mean error test: "+str(np.mean(err)))

# Create plots
n = 100
X,Y = np.meshgrid(np.linspace(0.,1.,100),np.linspace(0.,1.,100))

if usePlotly:
    from tfc.utils.PlotlyMakePlot import MakePlot

    p = MakePlot(r'x',r'y',zlabs=r'u(x,y,0.5)')
    p.Surface(x=X,
              y=Y,
              z=real(X,Y,0.5*np.ones_like(X)),
              showscale=False)
    p.view(azimuth=45,elevation=40)
    p.fig['layout']['scene']['aspectmode']='cube'
    p.show()

else:
    from matplotlib import cm
    from tfc.utils import MakePlot

    p = MakePlot(r'$x$',r'$y$',zlabs=r'$u(x,y,0.5)$')
    p.ax[0].plot_surface(X,Y,real(X,Y,0.5*np.ones_like(X)),cmap=cm.gist_rainbow,antialiased=False,rcount=n,ccount=n)
    p.ax[0].xaxis.labelpad = 20
    p.ax[0].yaxis.labelpad = 20
    p.ax[0].zaxis.labelpad = 20
    p.PartScreen(8,7)
    p.show()
