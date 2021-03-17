# This script solves the Lane-Emden equation (Section 3.8.1) in the TFC book
# Updated: 17 Mar 2021
####################################################################################################
# Differential Equation
#   y'' + 2/x y' + y^k = 0      where   k > 0 and x > 0
#
#   subject to: y(0) = 1, y'(0) = 0
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
a = 0   # specific problem type, a >=0 (analytical solution known for a = 0, 1, and 5)
xspan = [0., 10.] # problem domain range [x0, xf], where x0 > 0

## problem initial conditions: *********************************************************************
y0  = 1.  # y(x0)  = 1
y0p = 0.  # y'(x0) = 0
nC  = 2   # number of constraints

## construct univariate tfc class: *****************************************************************
tfc = utfc(N,nC,int(m),basis = basis, x0=xspan[0], xf=xspan[1])
x = tfc.x

H = tfc.H
dH = tfc.dH
H0 = H(x[0])
H0p = dH(x[0])

## define tfc constrained expression and derivatives: **********************************************
phi1 = lambda x: np.ones_like(x)
phi2 = lambda x: x

y = lambda x,xi: np.dot(H(x),xi) + phi1(x)*(y0  - np.dot(H0,xi)) + phi2(x)*(y0p - np.dot(H0p,xi))
yp = egrad(y)
ypp = egrad(yp)

## define the loss function: ***********************************************************************
L = lambda xi: x*ypp(x,xi) + 2.*yp(x,xi) + x*y(x,xi)**a

## solve the problem via nonlinear least-squares ***************************************************
xi = np.zeros(H(x).shape[1])

# if a==0 or a == 1, the problem is linear
if a == 0 or a == 1:
    xi,time = LS(xi,L,timer=True)
    iter = 1

else:
    xi,iter,time = NLLS(xi,L,timer=True)

## compute the error (if a = 0, 1, or 5): **********************************************************
if a == 0:
    ytrue = 1. - 1./6. * x**2
elif a == 1:
    ytrue = onp.ones_like(x)
    ytrue[1:] = np.sin(x[1:]) / x[1:]
elif a == 5:
    ytrue = (1. + x**2/3)**(-1/2)
else:
    ytrue = np.empty_like(x)

err = np.abs(y(x,xi) - ytrue)

## compute the residual of the loss vector: ********************************************************
res = np.abs(L(xi))

## plotting: ***************************************************************************************

# figure 1: solution
p1 = MakePlot(r'$x$',r'$y(x)$')
p1.ax[0].plot(x,y(x,xi))
p1.ax[0].grid(True)
p1.PartScreen(7.,6.)
p1.show()

# figure 2: residual
p2 = MakePlot(r'$x$',r'$|L(\xi)|$')
p2.ax[0].plot(x,res,'*')
p2.ax[0].grid(True)
p2.ax[0].set_yscale('log')
p2.PartScreen(7.,6.)
p2.show()

# figure 3: error (if a = 0, 1, or 5)
if a == 0 or a == 1 or a == 5:
    p3 = MakePlot(r'$x$',r'$|y_{true} - y(x)|$')
    p3.ax[0].plot(x,err,'*')
    p3.ax[0].grid(True)
    p3.ax[0].set_yscale('log')
    p3.PartScreen(7.,6.)
    p3.show()