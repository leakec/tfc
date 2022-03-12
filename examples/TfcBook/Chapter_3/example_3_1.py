# This script solves Problem #1 of Section 3's exercises in the TFC book
# Updated: 18 Mar 2021
####################################################################################################
# Differential Equation
#   yₓₓ + w^2 y = 0      where   w is the period
#
#   subject to: y(0) = 1, yₓ(0) = 0
####################################################################################################
from tfc import utfc
from tfc.utils import LsClass, egrad, MakePlot
from jax import jit
import jax.numpy as np

import numpy as onp
import tqdm
####################################################################################################

## user defined parameters: ************************************************************************
N = 100 # number of discretization points per TFC step
m = 40  # number of basis function terms
basis = 'CP' # basis function type

xspan = [0., 2.] # time range of problem
Nstep = int(xspan[1]/2) # number of TFC steps

y0  = 1.  # y(x0)  = 1
y0p = 0.  # yₓ(x0) = 0

w = 2.*np.pi

## problem initial conditions: *********************************************************************
if basis == 'CP' or 'LeP':
    nC  = 2   
elif basis == 'FS':
    nC  = 1
else:
    nC  = 0
# number of constraints

# length of time for one TFC step
xstep = (xspan[1]-xspan[0])/Nstep 
# !!! since this differential equation is not a explicit function of position 'x', I can get 
#     away with contructing the tfc class such that x = [0, xstep] an imposing a constant step so
#     that the mapping parameter c = (zf-z0)/(xf-x0) is also constant


## construct univariate tfc class: *****************************************************************
tfc = utfc(N+1, nC, int(m+1), basis = basis, x0=0, xf=xstep)
x = tfc.x
# !!! notice I am using N+1 for the number of points. this is because I will be using the last point
#     of a segment 'n' for the initial conditons of the 'n+1' segment

H = tfc.H
dH = tfc.dH
H0 = H(x[0:1])
H0p = dH(x[0:1])

## define tfc constrained expression and derivatives: **********************************************
# switching function
phi1 = lambda x: np.ones_like(x)
phi2 = lambda x: x

# tfc constrained expression
y = lambda x,xi,IC: np.dot(H(x),xi) + phi1(x)*(IC['y0']  - np.dot(H0,xi)) \
                                    + phi2(x)*(IC['y0p'] - np.dot(H0p,xi))
# !!! notice here that the initial conditions are passed as a dictionary (i.e. IC['y0'])
#     this will be important so that the least-squares does not need to be re-JITed   

yp = egrad(y)
ypp = egrad(yp)

## define the loss function: ***********************************************************************
#   yₓₓ + w^2 y = 0
L = jit(lambda xi,x,IC: ypp(x,xi,IC) + w**2*y(x,xi,IC))

## construct the least-squares class: **************************************************************
xi0 = np.zeros(H(x).shape[1])
IC = {'y0': np.array([y0]), 'y0p': np.array([y0p])}


ls = LsClass(xi0,L,x,IC,timer=True,constant_arg_nums=[1])

## initialize dictionary to record solution: *******************************************************
xSol    = onp.zeros((Nstep,N))
ySol    = onp.zeros_like(xSol)  
res     = onp.zeros_like(xSol)
err     = onp.zeros_like(xSol)
time    = onp.zeros(Nstep)

xSol[0,:] = x[:-1]
xFinal = x[-1]
## 'propagation' loop: *****************************************************************************
for i in tqdm.trange(Nstep):
    xi, time[i] = ls.run(xi0,IC)

    # print solution to dictionary
    if i > 0:
        xSol[i,:]    = xFinal + x[:-1]
        xFinal += x[-1]

    # save solution to python dictionary 
    ySol[i,:]   = y(x,xi,IC)[:-1]
    res[i,:]    = np.abs(L(xi,x,IC))[:-1]

    # update initial condtions
    IC['y0']  = y(x,xi,IC)[-1]
    IC['y0p'] = yp(x,xi,IC)[-1]

## compute the error: ******************************************************************************
A = np.sqrt(y0**2 + (y0p/w)**2)
Phi = np.arctan(-y0p/w/y0)
yTrue = A*np.cos(w*xSol+Phi)

err = np.abs(ySol-yTrue)

## print status of run: ****************************************************************************
print('TFC least-squares time[s]: ' +'\t'+ str((time.sum())))
print('Max residual:' +'\t'*3+ str(res.max()))
print('Max error:' +'\t'*3+ str(err.max()))

## plotting: ***************************************************************************************
# figure 1: solution
p1 = MakePlot(r'$x$',r'$y(t)$')
p1.ax[0].plot(xSol.flatten(),ySol.flatten())
p1.ax[0].grid(True)
p1.PartScreen(7.,6.)
p1.show()

# figure 2: residual
p2 = MakePlot(r'$t$',r'$|L(\xi)|$')
p2.ax[0].plot(xSol.flatten(),res.flatten(),'*')
p2.ax[0].grid(True)
p2.ax[0].set_yscale('log')
p2.PartScreen(7.,6.)
p2.show()

# figure 3: error
p3 = MakePlot(r'$t$',r'$|y_{true} - y(t)|$')
p3.ax[0].plot(xSol.flatten(),err.flatten(),'*')
p3.ax[0].grid(True)
p3.ax[0].set_yscale('log')
p3.PartScreen(7.,6.)
p3.show()
