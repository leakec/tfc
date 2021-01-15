# Import python packages
import numpy as onp
import numpy.matlib
import jax.numpy as np
from time import process_time as timer
from matplotlib import cm

# Import TFC classes
from tfc import mtfc
from tfc.utils import egrad, TFCDict, NLLS, MakePlot

# Constants:
Hb = 1.0
tf = 3.0
xend = 15.0
x0 = np.array([0.,-Hb/2.,0.])
xf = np.array([xend,Hb/2.,tf])

rho = 1.
mu = 1.
P = -5.

xTfc = True # Change to True to use X-TFC instead

# TFC parameters:
n = 10

if xTfc:
    m = 200
    basis = 'ELMTanh'
    nC = -1
else:
    m = 10
    basis = 'CP'
    nC = [2,2,1]

maxIter = 50

# Create the TFC Class:
N = [n,n,n]
myTfc = mtfc(N,nC,m,dim=3,basis=basis,x0=x0,xf=xf)
x = myTfc.x

# Get the basis functions
H = myTfc.H
Hx = myTfc.Hx

# Create the TFC constrained expression (here f stands as a placeholder for u and v)
f1 = lambda xi,*x: np.dot(H(*x),xi)-np.dot(H(np.zeros_like(x[0]),x[1],x[2]),xi)-x[0]*np.dot(Hx(xend*np.ones_like(x[0]),x[1],x[2]),xi)
f2 = lambda xi,*x: f1(xi,*x)-(Hb-2.*x[1])/(2.*Hb)*f1(xi,x[0],-Hb/2.*np.ones_like(x[1]),x[2])-(Hb+2.*x[1])/(2.*Hb)*f1(xi,x[0],Hb/2.*np.ones_like(x[1]),x[2])
f = lambda xi,*x: f2(xi,*x)-f2(xi,x[0],x[1],np.zeros_like(x[2]))

fx = egrad(f,1)
f2x = egrad(fx,1)
fy = egrad(f,2)
f2y = egrad(fy,2)
ft = egrad(f,3)

# Create the residual and jacobian
L1 = lambda xiu,xiv,*x: fx(xiu,*x)+fy(xiv,*x)
L2 = lambda xiu,xiv,*x: rho*(ft(xiu,*x)+f(xiu,*x)*fx(xiu,*x)+f(xiv,*x)*fy(xiu,*x))+P-mu*(f2x(xiu,*x)+f2y(xiu,*x))
L3 = lambda xiu,xiv,*x: rho*(ft(xiv,*x)+f(xiu,*x)*fx(xiv,*x)+f(xiv,*x)*fy(xiv,*x))-mu*(f2x(xiv,*x)+f2y(xiv,*x))
L = lambda xi: np.hstack([L1(xi['xiu'],xi['xiv'],*x),L2(xi['xiu'],xi['xiv'],*x),L3(xi['xiu'],xi['xiv'],*x)])

# Calculate the xi values
M = H(*x).shape[1]
xiu = np.zeros(M)
xiv = np.zeros(M)
xi = TFCDict({'xiu':xiu,'xiv':xiv})

if xTfc: 
    xi,it,time = NLLS(xi,L,maxIter=maxIter,method='lstsq',timer=True)
else:
    xi,it,time = NLLS(xi,L,maxIter=maxIter,method='pinv',timer=True)
xiu = xi['xiu']; xiv = xi['xiv']

# Calcualte u and plot for different times
n = 100
X = onp.matlib.repmat(onp.reshape(onp.linspace(0,xf[0],num=n),(n,1)),n,1).flatten()
Y = onp.reshape(onp.matlib.repmat(onp.reshape(onp.linspace(-Hb/2.,Hb/2.,num=n),(n,1)),1,n),(n**2,1)).flatten()
xTest = onp.zeros((3,n**2*3))
xTest[0,:] = onp.hstack([X,]*3)
xTest[1,:] = onp.hstack([Y,]*3)
xTest[2,:] = onp.hstack([onp.ones(n**2)*0.01,onp.ones(n**2)*0.1,onp.ones(n**2)*tf])

p = []; U = [];
vals = [0.01,0.1,tf]
u = f(xiu,*xTest)
for k in range(len(vals)):
    p.append(MakePlot(r'$x (m)$',r'$y (m)$'))
    ind = np.where(onp.round(xTest[2],12)==onp.round(vals[k],12))
    U.append(np.reshape(u[ind],(n,n)))

Xm = np.reshape(xTest[0][ind],(n,n))
Ym = np.reshape(xTest[1][ind],(n,n))

dark = np.block(U)
vMin = np.min(dark)
vMax = np.max(dark)
def MakeContourPlot(Xm,Ym,Um):
    p = MakePlot(r'$x$ (m)',r'$y$ (m)')
    C = p.ax[0].contourf(Xm,Ym,Um,vmin=vMin,vmax=vMax,cmap=cm.gist_rainbow)
    cbar = p.fig.colorbar(C)
    return p

plots = [MakeContourPlot(Xm,Ym,U[0]),MakeContourPlot(Xm,Ym,U[1]),MakeContourPlot(Xm,Ym,U[2])]
for k,j in enumerate(plots):
    j.FullScreen()
    j.show()
    j.save('TFC'+str(k),fileType='png')

# U error
ind = np.where(xTest[2]==tf)
ind2 = np.where(xTest[0][ind] == xend)
uEnd = u[ind][ind2]
y = xTest[1][ind][ind2]
uTrue = P*(4.*y**2-Hb**2)/(8.*mu)
uErr = np.abs(uEnd-uTrue)
print("Max u error at the end: "+str(np.max(uErr)))
print("Mean u error at the end: "+str(np.mean(uErr)))

# V error
vEnd = f(xiv,*xTest)[ind][ind2]
vTrue = np.zeros_like(vEnd)
vErr = np.abs(vEnd-vTrue)
print("Max v error at the end: "+str(np.max(vErr)))
print("Mean v error at the end: "+str(np.mean(vErr)))
