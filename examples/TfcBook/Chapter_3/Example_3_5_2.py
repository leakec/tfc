# This script solves the Lane-Emden equation (Section 3.8.1) in the TFC book
# Updated: 17 Mar 2021
####################################################################################################
# Differential Equation
#   y'' + yy' = exp(-2x) sin(x) [cos(x) -sin(x)] - 2exp(-x)cos(x)
#
#   subject to: y(0) = 0, y(pi) = 0
####################################################################################################
from tfc import utfc
from tfc.utils import egrad, NLLS, LS, MakePlot
import jax.numpy as np

import numpy as onp
####################################################################################################

## user defined parameters: ************************************************************************
N = 100 # number of discretization points
m = 60  # number of basis function terms
basis = 'CP' # basis function type

## problem boundary conditions: ********************************************************************
xspan = [0., np.pi]
y0  = 0. # y(0)  = 0
yf  = 0. # y(pi) = 0
nC  = 2   # number of constraints

## construct univariate tfc class: *****************************************************************
tfc = utfc(N, nC, int(m), basis = basis, x0=xspan[0], xf=xspan[1])
x = tfc.x

H = tfc.H
H0 = H(tfc.x[0])
Hf = H(tfc.x[-1])

## define tfc constrained expression and derivatives: **********************************************
# switching function
phi1 = lambda x: (np.pi - x)/np.pi
phi2 = lambda x: x/np.pi

# forcing term
f = lambda x: np.exp(-2.*x) * np.sin(x) * (np.cos(x) - np.sin(x)) - 2.*np.exp(-x)*np.cos(x)

# tfc constrained expression
y = lambda x,xi: np.dot(H(x),xi) + phi1(x)*(y0 - np.dot(H0,xi)) + phi2(x)*(yf - np.dot(Hf,xi))
yp = egrad(y)
ypp = egrad(yp)

## define the loss function: ***********************************************************************
L = lambda xi: ypp(x,xi) + y(x,xi)*yp(x,xi) - f(x)

## solve the problem via nonlinear least-squares ***************************************************
xi = np.zeros(H(x).shape[1])

xi,iter,time = NLLS(xi,L,timer=True)

## compute the error: ******************************************************************************
ytrue = np.exp(-x) * np.sin(x)

err = np.abs(y(x,xi) - ytrue)

## compute the residual of the loss vector: ********************************************************
res = np.abs(L(xi))

## plotting: ***************************************************************************************

# figure 1: solution
p1 = MakePlot(r'$x$',r'$y(x)$')
p1.ax[0].plot(x,y(x,xi))
p1.ax[0].grid(True)
p1.ax[0].set_xlim(xspan[0], xspan[1])
p1.PartScreen(7.,6.)
p1.show()

# figure 2: residual
p2 = MakePlot(r'$x$',r'$|L(\xi)|$')
p2.ax[0].plot(x,res,'*')
p2.ax[0].grid(True)
p2.ax[0].set_yscale('log')
p2.ax[0].set_xlim(xspan[0], xspan[1])
p2.PartScreen(7.,6.)
p2.show()

# figure 3: error
p3 = MakePlot(r'$x$',r'$|y_{true} - y(x)|$')
p3.ax[0].plot(x,err,'*')
p3.ax[0].grid(True)
p3.ax[0].set_yscale('log')
p3.ax[0].set_xlim(xspan[0], xspan[1])
p3.PartScreen(7.,6.)
p3.show()