# This script solves Problem #1 of Chapter 1's exercises in the TFC book
####################################################################################################
# Differential Equation
#   yₓₓ + w^2 y = 0      where   w is the period
#
#   subject to: y(0) = 1, yₓ(0) = 0
####################################################################################################
from tfc import utfc
from tfc.utils import MakePlot, step
import jax.numpy as np

import numpy as onp
####################################################################################################

## user defined parameters: ************************************************************************
N = 100 # number of discretization points per TFC step
m = 5  # number of basis function terms
basis = 'CP' # basis function type
nC = 2  # number of constraints

## problem initial conditions: *********************************************************************
tspan = [-1., 1.] # time range of problem

initial = np.array([-1.0, -1.0])
final  = np.array([ 1.0,  1.0])

Nlines = 5

## keep-out parameters: ****************************************************************************
xbound = np.array([-0.25, 0.25])
ybound = np.array([-0.25, 0.25])

## construct univariate tfc class: *****************************************************************
tfc = utfc(N, nC, m, basis = basis, x0=tspan[0], xf=tspan[-1])

t = tfc.x
H = tfc.H
H0 = H(t[0])
Hf = H(t[-1])

## define tfc constrained expression and derivatives: **********************************************
# switching function
phi1 = lambda t: (t[-1] - t) / (t[-1] - t[0])
phi2 = lambda t: (t - t[0])  / (t[-1] - t[0])

# tfc constrained expression (without inequality constraints)
xhat = lambda xi: np.dot(H(t),xi)  + phi1(t)*(initial[0] - np.dot(H0,xi)) \
                                   + phi2(t)*(final[0]   - np.dot(Hf,xi))
yhat = lambda xi: np.dot(H(t),xi)  + phi1(t)*(initial[1] - np.dot(H0,xi)) \
                                   + phi2(t)*(final[1]   - np.dot(Hf,xi)) 

# construct pseudo-switching functions for the box constraints
Phi1 = lambda ghat, bound: step(bound[1] - ghat)
Phi2 = lambda ghat, bound: step(ghat - bound[0])
Phi3 = lambda ghat, bound: step((bound[1]+bound[0])/2. - ghat)

# tfc constrained expression (with inequality constraints)
x = lambda xi: xhat(xi) \
            +  (xbound[0]-xhat(xi))*(Phi1(yhat(xi),ybound)*Phi2(yhat(xi),ybound) *\
                                     Phi3(xhat(xi),xbound)*Phi2(xhat(xi),xbound)) \
            +  (xbound[1]-xhat(xi))*(Phi1(yhat(xi),ybound)*Phi2(yhat(xi),ybound) *\
                                     Phi3(-xhat(xi),-xbound)*Phi1(xhat(xi),xbound))

y = lambda xi: yhat(xi) \
             + (ybound[0]-yhat(xi))*(Phi1(xhat(xi),xbound)*Phi2(xhat(xi),xbound) *\
                                     Phi3(yhat(xi),ybound)*Phi2(yhat(xi),ybound)) \
             + (ybound[1]-yhat(xi))*(Phi1(xhat(xi),xbound)*Phi2(xhat(xi),xbound) *\
                                     Phi3(-yhat(xi),-ybound)*Phi1(yhat(xi),ybound))


x_xi = 1./3. * onp.random.randn(H(t).shape[1], Nlines)
y_xi = 1./3. * onp.random.randn(H(t).shape[1], Nlines)

## plotting: ***************************************************************************************
# figure 1: solution
p1 = MakePlot(r'$x(t)$',r'$y(t)$')
for i in range(Nlines):
        p1.ax[0].plot(x(x_xi[:,i]), y(y_xi[:,i]))
p1.ax[0].grid(True)
p1.PartScreen(7.,6.)
p1.show()

# # figure 2: residual
# p2 = MakePlot(r'$t$',r'$|L(\xi)|$')
# p2.ax[0].plot(xSol.flatten(),res.flatten(),'*')
# p2.ax[0].grid(True)
# p2.ax[0].set_yscale('log')
# p2.PartScreen(7.,6.)
# p2.show()

# # figure 3: error
# p3 = MakePlot(r'$t$',r'$|y_{true} - y(t)|$')
# p3.ax[0].plot(xSol.flatten(),err.flatten(),'*')
# p3.ax[0].grid(True)
# p3.ax[0].set_yscale('log')
# p3.PartScreen(7.,6.)
# p3.show()
