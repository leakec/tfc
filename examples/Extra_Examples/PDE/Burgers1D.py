import tqdm

import numpy as onp
import jax.numpy as np

from tfc import mtfc
from tfc.utils import NLLS, egrad

# Constants and switches:
c = 1.
nu = 0.5
xInit = -3.
xFinal = 3.
alpha = 1.

x0 = np.array([xInit,0.])
xf = np.array([xFinal,1.])

n = 30 
nTest = 100
m = 600

usePlotly = True

# Real analytical solution:
real = lambda x,t: c/alpha-c/alpha*np.tanh(c*(x-c*t)/(2.*nu))

# Create the mtfc class:
N = [n,n]
nC = -1
tfc = mtfc(N,nC,m,dim=2,basis='ELMTanh',x0=x0,xf=xf)
x = tfc.x

# Get the basis functions
H = tfc.H

# Create the TFC constrained expression
u1 = lambda xi,*x: np.dot(H(*x),xi)+\
        (xFinal-x[0])/(xFinal-xInit)*(c/alpha-c/alpha*np.tanh(c*(xInit-c*x[1])/(2.*nu))-np.dot(H(xInit*np.ones_like(x[0]),x[1]),xi))+\
        (x[0]-xInit)/(xFinal-xInit)*(c/alpha-c/alpha*np.tanh(c*(xFinal-c*x[1])/(2.*nu))-np.dot(H(xFinal*np.ones_like(x[0]),x[1]),xi))
u = lambda xi,*x: u1(xi,*x)+\
        c/alpha-c/alpha*np.tanh(c*x[0]/(2.*nu))-u1(xi,x[0],np.zeros_like(x[1]))

# Create the residual
ux = egrad(u,1)
d2x = egrad(ux,1)
ut = egrad(u,2)
r = lambda xi: ut(xi,*x)+alpha*u(xi,*x)*ux(xi,*x)-nu*d2x(xi,*x)

# Solve the problem
xi = np.zeros(H(*x).shape[1])
xi,it,time = NLLS(xi,r,method='lstsq',timer=True,timerType="perf_counter")

# Calculate error at the test points:
dark = np.meshgrid(np.linspace(xInit,xFinal,nTest),np.linspace(0.,1.,nTest))
xTest = tuple([j.flatten() for j in dark])
err = np.abs(u(xi,*xTest)-real(*xTest))

print("Training time: "+str(time))
print("Max error: "+str(np.max(err)))
print("Mean error: "+str(np.mean(err)))

# Plot analytical solution
if usePlotly:
    from tfc.utils.PlotlyMakePlot import MakePlot

    p = MakePlot(r"x",r"t",zlabs=r"u(x,t)")
    p.Surface(x=xTest[0].reshape((nTest,nTest)),
              y=xTest[1].reshape((nTest,nTest)),
              z=real(*xTest).reshape((nTest,nTest)),
              showscale=False)
    p.PartScreen(9,8)
    p.show()

else:
    from matplotlib import cm
    from MakePlot import MakePlot

    p = MakePlot(r"$x$",r"$t$",zlabs=r"$u(x,t)$")
    p.ax[0].plot_surface(xTest[0].reshape((nTest,nTest)),
                         xTest[1].reshape((nTest,nTest)),
                         real(*xTest).reshape((nTest,nTest)),
                         cmap=cm.gist_rainbow)
    p.ax[0].xaxis.labelpad = 10
    p.ax[0].yaxis.labelpad = 20
    p.ax[0].zaxis.labelpad = 10
    p.show()
