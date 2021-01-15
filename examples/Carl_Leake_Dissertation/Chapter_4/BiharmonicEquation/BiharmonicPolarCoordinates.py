import tqdm

import numpy as onp
import jax.numpy as np
from matplotlib import cm

from tfc import mtfc
from tfc.utils import LS, egrad, MakePlot

# Constants and switches:
x0 = np.array([1.,0.])
xf = np.array([4.,2.*np.pi])

n = 30
nTest = 100

xTFC = False # Set to True to use X-TFC rather than TFC

# Real analytical solution:
real = lambda r,th: r**2/8.+np.pi*np.cos(th)/r + r**2/4.*np.sin(2.*th)+r**3/16.*np.sin(3.*th)
d2realdr2 = egrad(egrad(real,0),0)

# MC constants:
if xTFC:
    m = 335
else:
    m = 30

# Create the TFC Class:
N = [n,n]
if xTFC:
    nC = -1
    myTfc = mtfc(N,nC,m,dim=2,basis='ELMTanh',x0=x0,xf=xf)
else:
    nC = [4,4]
    myTfc = mtfc(N,nC,m,dim=2,basis='CP',x0=x0,xf=xf)
x = myTfc.x

# Get the basis functions
H = myTfc.H
Hrr = myTfc.Hx2

# Create the TFC constrained expression
u1 = lambda xi,*x: np.dot(H(*x),xi)+\
        (-x[0]+4.)/3.*(real(np.ones_like(x[0]),x[1])-np.dot(H(np.ones_like(x[0]),x[1]),xi))+\
        (x[0]-1.)/3.*(real(4.*np.ones_like(x[0]),x[1])-np.dot(H(4.*np.ones_like(x[0]),x[1]),xi))+\
        (-x[0]**3+12.*x[0]**2-39.*x[0]+28.)/18.*(d2realdr2(np.ones_like(x[0]),x[1])-np.dot(Hrr(np.ones_like(x[0]),x[1]),xi))+\
        (x[0]**3-3.*x[0]**2-6.*x[0]+8.)/18.*(d2realdr2(4.*np.ones_like(x[0]),x[1])-np.dot(Hrr(4.*np.ones_like(x[0]),x[1]),xi))
u1th = egrad(u1,2)
u12th = egrad(u1th,2)
u13th = egrad(u12th,2)
u = lambda xi,*x: u1(xi,*x)-\
        x[1]/(2.*np.pi)*(u1(xi,x[0],2.*np.pi*np.ones_like(x[1]))-u1(xi,x[0],np.zeros_like(x[1])))+\
        (-x[1]**2+2.*np.pi*x[1])/(4.*np.pi)*(u1th(xi,x[0],2.*np.pi*np.ones_like(x[1]))-u1th(xi,x[0],np.zeros_like(x[1])))+\
        (-x[1]**3+3.*np.pi*x[1]**2-2.*np.pi**2*x[1])/(12.*np.pi)*(u12th(xi,x[0],2.*np.pi*np.ones_like(x[1]))-u12th(xi,x[0],np.zeros_like(x[1])))+\
        (-x[1]**4+4.*np.pi*x[1]**3-4.*np.pi**2*x[1]**2)/(48.*np.pi)*(u13th(xi,x[0],2.*np.pi*np.ones_like(x[1]))-u13th(xi,x[0],np.zeros_like(x[1])))
        
# Create the residual
ur = egrad(u,1)
u2r = egrad(ur,1)
u3r = egrad(u2r,1)
u4r = egrad(u3r,1)

ur2th = egrad(egrad(ur,2),2)
u2th = egrad(egrad(u,2),2)
u2r2th = egrad(egrad(u2r,2),2)
u4th = egrad(egrad(u2th,2),2)

L = lambda xi: u4r(xi,*x)+2./x[0]**2*u2r2th(xi,*x)+1./x[0]**4*u4th(xi,*x)+2./x[0]*u3r(xi,*x)-2./x[0]**3*ur2th(xi,*x)-1./x[0]**2*u2r(xi,*x)+4./x[0]**4*u2th(xi,*x)+1./x[0]**3*ur(xi,*x)

# Solve the problem
xi = np.zeros(H(*x).shape[1])
if xTFC:
    xi,time = LS(xi,L,method='lstsq',timer=True)
else:
    xi,time = LS(xi,L,timer=True)

# Calculate error at the test points:
dark = np.meshgrid(np.linspace(x0[0],xf[0],nTest),np.linspace(x0[1],xf[1],nTest))
xTest = tuple([j.flatten() for j in dark])
err = np.abs(u(xi,*xTest)-real(*xTest))
print("Time: "+str(time))
print("Max Error: "+str(np.max(err)))
print("Mean Error: "+str(np.mean(err)))

# Plot the analytical solution in polar coordinates
X = xTest[0]*np.cos(xTest[1])
Y = xTest[0]*np.sin(xTest[1])

xlabs = [[r'$x$',r'$x$'],['',r'$x$']]
ylabs = [[r'',r'$y$'],[r'$y$',r'$y$']]
zlabs = [[r'$u(x,y)$',r''],[r'$u(x,y)$',r'$u(x,y)$']]

azim = [-90,-90,0,-135]
elev = [0,90,0,45]

p = MakePlot(xlabs,ylabs,zlabs=zlabs)
for k in range(4):
    p.ax[k].plot_surface(X.reshape((nTest,nTest)),Y.reshape((nTest,nTest)),real(*xTest).reshape((nTest,nTest)),cmap=cm.nipy_spectral)
    p.ax[k].xaxis.labelpad = 15
    p.ax[k].yaxis.labelpad = 15
    p.ax[k].zaxis.labelpad = 10
    p.ax[k].view_init(azim=azim[k],elev=elev[k])
    if not k == 4:
        p.ax[k].set_proj_type('ortho')

p.ax[1].tick_params(axis='both', which='major', pad=8)
p.ax[1].xaxis.labelpad = 10
p.ax[1].yaxis.labelpad = 10

p.ax[0].set_yticklabels([])
p.ax[1].set_zticklabels([])
p.ax[2].set_xticklabels([])
p.fig.subplots_adjust(wspace=0.05, hspace=0.05)
p.PartScreen(10,9)
p.show()

# Plot the error in polar coordinates
p1 = MakePlot(r'$r$',r'$\theta$',zlabs=r'error')
p1.ax[0].plot_surface(xTest[0].reshape((nTest,nTest)),xTest[1].reshape((nTest,nTest)),err.reshape((nTest,nTest)),cmap=cm.gist_rainbow)
p1.ax[0].xaxis.labelpad = 20
p1.ax[0].yaxis.labelpad = 20
p1.ax[0].zaxis.labelpad = 20
p1.FullScreen()
p1.show()
