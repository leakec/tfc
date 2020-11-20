import tqdm
from time import process_time as timer

import numpy as onp
import jax.numpy as np
from matplotlib import cm

from tfc import mtfc
from tfc.utils import LS, egrad, MakePlot

# Constants and switches:
x0 = np.array([0.,0.])
xf = np.array([1.,1.])

n = 20
nTest = 100

xTFC = False # Set to True to use X-TFC rather than TFC

# Real analytical solution:
real = lambda x,y: 1./np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)

# MC constants:
if xTFC:
    m = 335
else:
    m = 26

# Create the TFC Class:
N = [n,n]
if xTFC:
    nC = [-1,]
    myTfc = mtfc(N,nC,m,dim=2,basis='ELMTanh',x0=x0,xf=xf)
else:
    nC = [4,4]
    myTfc = mtfc(N,nC,m,dim=2,basis='CP',x0=x0,xf=xf)
x = myTfc.x

# Get the basis functions
H = myTfc.H
Hxx = myTfc.Hx2

# Create the TFC constrained expression
u1 = lambda xi,*x: np.dot(H(*x),xi)-\
        (1.-x[0])*np.dot(H(np.zeros_like(x[0]),x[1]),xi)-\
        x[0]*np.dot(H(np.ones_like(x[0]),x[1]),xi)-\
        (-x[0]**3+3*x[0]**2-2.*x[0])/6.*np.dot(Hxx(np.zeros_like(x[0]),x[1]),xi)-\
        (x[0]**3-x[0])/6.*np.dot(Hxx(np.ones_like(x[0]),x[1]),xi)
u1yy = egrad(egrad(u1,2),2)
u = lambda xi,*x: u1(xi,*x)-\
        (1.-x[1])*u1(xi,x[0],np.zeros_like(x[1]))-\
        x[1]*u1(xi,x[0],np.ones_like(x[1]))-\
        (-x[1]**3+3*x[1]**2-2.*x[1])/6.*u1yy(xi,x[0],np.zeros_like(x[1]))-\
        (x[1]**3-x[1])/6.*u1yy(xi,x[0],np.ones_like(x[1]))
        
        
# Create the residual
u4x = egrad(egrad(egrad(egrad(u,1),1),1),1)
u4y = egrad(egrad(egrad(egrad(u,2),2),2),2)
u2x2y = egrad(egrad(egrad(egrad(u,1),1),2),2)
L = lambda xi: u4x(xi,*x)+u4y(xi,*x)+2.*u2x2y(xi,*x)-4.*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])

# Solve the problem
xi = np.zeros(H(*x).shape[1])
if xTFC:
    xi,time = LS(xi,L,method='lstsq',timer=True)
else:
    xi,time = LS(xi,L,timer=True)

# Calculate error at the test points:
dark = np.meshgrid(np.linspace(0,1,nTest),np.linspace(0,1,nTest))
xTest = tuple([j.flatten() for j in dark])
err = np.abs(u(xi,*xTest)-real(*xTest))
print("Time: "+str(time))
print("Max Error: "+str(np.max(err)))
print("Mean Error: "+str(np.mean(err)))

# Plot the analytical solution
p = MakePlot(r'$x$',r'$y$',zlabs=r'$u(x,y)$')
p.ax[0].plot_surface(xTest[0].reshape((nTest,nTest)),xTest[1].reshape((nTest,nTest)),real(*xTest).reshape((nTest,nTest)),cmap=cm.gist_rainbow)
p.ax[0].xaxis.labelpad = 20
p.ax[0].yaxis.labelpad = 20
p.ax[0].zaxis.labelpad = 20
p.FullScreen()
p.show()
