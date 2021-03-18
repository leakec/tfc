# This script solves Problem #1 of Section 3's exercises in the TFC book
# Updated: 18 Mar 2021
####################################################################################################
# Differential Equation
#   y'' + w^2 y = 0      where   w is the period
#
#   subject to: y(0) = y0, y'(0) = y'0
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

tspan = [0., 500.] # time range of problem
Nstep = int(tspan[1]/2) # number of TFC steps

y0  = 1.  # y(t0)  = y0
y0d = 0.  # y'(t0) = y'0

w = 2.*np.pi

## problem initial conditions: *********************************************************************
nC  = 2   # number of constraints

# length of time for one TFC step
tstep = (tspan[1]-tspan[0])/Nstep 
# !!! since this differential equation is not a explicit function of time 't', I can get away with
#     contructing the tfc class such that t = [0, tstep] an imposing a constant step so that the
#     mapping parameter c = (zf-z0)/(tf-t0) is also constant


## construct univariate tfc class: *****************************************************************
tfc = utfc(N+1, nC, int(m+1), basis = basis, x0=0, xf=tstep)
t = tfc.x
# !!! notice I am using N+1 for the number of points. this is because I will be using the last point
#     of a segment 'n' for the initial conditons of the 'n+1' segment

H = tfc.H
dH = tfc.dH
H0 = H(t[0])
H0p = dH(t[0])

## define tfc constrained expression and derivatives: **********************************************
# switching function
phi1 = lambda t: np.ones_like(t)
phi2 = lambda t: t

# tfc constrained expression
y = lambda t,xi,IC: np.dot(H(t),xi) + phi1(t)*(IC['y0']  - np.dot(H0,xi)) \
                                    + phi2(t)*(IC['y0d'] - np.dot(H0p,xi))
# !!! notice here that the initial conditions are passed as a dictionary (i.e. IC['y0'])
#     this will be important so that the least-squares does not need to be re-JITed   

yd = egrad(y)
ydd = egrad(yd)

## define the loss function: ***********************************************************************
L = jit(lambda xi,IC: ydd(t,xi,IC) + w**2*y(t,xi,IC))

## construct the least-squares class: **************************************************************
xi0 = np.zeros(H(t).shape[1])
IC = {'y0': np.array([y0]), 'y0d': np.array([y0d])}


ls = LsClass(xi0,L,timer=True)

## initialize dictionary to record solution: *******************************************************
sol = { 't'   : onp.zeros((Nstep,N)), 'y'  : onp.zeros((Nstep,N)), \
        'yd'  : onp.zeros((Nstep,N)), 'ydd': onp.zeros((Nstep,N)), \
        'res' : onp.zeros((Nstep,N)), 'err': onp.zeros((Nstep,N)), \
        'time': onp.zeros(Nstep)}

sol['t'][0,:] = t[:-1]
tFinal = t[-1]
## 'propagation' loop: *****************************************************************************
for i in tqdm.trange(Nstep):
    xi, sol['time'][i] = ls.run(xi0,IC)

    # print solution to dictionary
    if i > 0:
        sol['t'][i,:]    = tFinal + t[:-1]
        tFinal += t[-1]

    # save solution to python dictionary 
    sol['y'][i,:]    = y(t,xi,IC)[:-1]
    # sol['yd'][i,:]   = yd(t,xi,IC)[:-1]
    # sol['ydd'][i,:]  = ydd(t,xi,IC)[:-1]
    sol['res'][i,:]  = np.abs(L(xi,IC))[:-1]
    # !!! disclaimer, saving data to the dictionary drastically increases script run time

    # update initial condtions
    IC['y0']  = y(t,xi,IC)[-1]
    IC['y0d'] = yd(t,xi,IC)[-1]

## compute the error: ******************************************************************************
A = np.sqrt(y0**2 + (y0d/w)**2)
Phi = np.arctan(-y0d/w/y0)
ytrue = A*np.cos(w*sol['t']+Phi)

sol['err'] = np.abs(sol['y'] - ytrue)

## plotting: ***************************************************************************************

# figure 1: solution
p1 = MakePlot(r'$x$',r'$y(t)$')
p1.ax[0].plot(sol['t'].flatten(),sol['y'].flatten())
p1.ax[0].grid(True)
p1.PartScreen(7.,6.)
p1.show()

# figure 2: residual
p2 = MakePlot(r'$t$',r'$|L(\xi)|$')
p2.ax[0].plot(sol['t'].flatten(),sol['res'].flatten(),'*')
p2.ax[0].grid(True)
p2.ax[0].set_yscale('log')
p2.PartScreen(7.,6.)
p2.show()

# figure 3: error
p3 = MakePlot(r'$t$',r'$|y_{true} - y(t)|$')
p3.ax[0].plot(sol['t'].flatten(),sol['err'].flatten(),'*')
p3.ax[0].grid(True)
p3.ax[0].set_yscale('log')
p3.PartScreen(7.,6.)
p3.show()
