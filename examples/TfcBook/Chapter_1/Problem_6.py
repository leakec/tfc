# This script solves Problem #1 of Chapter 1's exercises in the TFC book
####################################################################################################
####################################################################################################
from tfc import utfc
from tfc.utils import MakePlot, step
import jax.numpy as np
from matplotlib.patches import Rectangle

import numpy as onp
####################################################################################################

## user defined parameters: ************************************************************************
N = 1000 # number of discretization points per TFC step
m = 10  # number of basis function terms
basis = 'CP' # basis function type
nC = 2  # number of constraints

## problem initial conditions: *********************************************************************
tspan = [0., 1.] # time range of problem

initial = np.array([-1.0, -1.0])
final  = np.array([ 1.0,  1.0])

Nlines = 20

## keep-out parameters: ****************************************************************************
xbound = np.array([-0.5, 0.5])
ybound = np.array([-0.5, 0.5])

## construct univariate tfc class: *****************************************************************
tfc = utfc(N, nC, m, basis = basis, x0=tspan[0], xf=tspan[-1])

t = tfc.x
H = tfc.H
H0 = H(t[0])
Hf = H(t[-1])

## define tfc constrained expression: **************************************************************
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
x = lambda x_xi, y_xi: xhat(x_xi) \
            +  (xbound[0]-xhat(x_xi))*(Phi1(yhat(y_xi),ybound)*Phi2(yhat(y_xi),ybound) *\
                                       Phi3(xhat(x_xi),xbound)*Phi2(xhat(x_xi),xbound)) \
            +  (xbound[1]-xhat(x_xi))*(Phi1(yhat(y_xi),ybound)*Phi2(yhat(y_xi),ybound) *\
                                       Phi3(-xhat(x_xi),-xbound)*Phi1(xhat(x_xi),xbound))

y = lambda x_xi, y_xi: yhat(y_xi) \
             + (ybound[0]-yhat(y_xi))*(Phi1(xhat(x_xi),xbound)*Phi2(xhat(x_xi),xbound) *\
                                       Phi3(yhat(y_xi),ybound)*Phi2(yhat(y_xi),ybound)) \
             + (ybound[1]-yhat(y_xi))*(Phi1(xhat(x_xi),xbound)*Phi2(xhat(x_xi),xbound) *\
                                       Phi3(-yhat(y_xi),-ybound)*Phi1(yhat(y_xi),ybound))

onp.random.seed(4) # fixes random seed to creat the same plot in book
x_xi = 0.1 * onp.random.randn(H(t).shape[1], Nlines)
y_xi = 0.1 * onp.random.randn(H(t).shape[1], Nlines)

## plotting: ***************************************************************************************
p1 = MakePlot(r'$x(t)$',r'$y(t)$')
for i in range(Nlines):
        p1.ax[0].plot(x(x_xi[:,i],y_xi[:,i]), y(x_xi[:,i],y_xi[:,i]))

p1.ax[0].add_patch(Rectangle((xbound[0],ybound[0]), xbound[1]-xbound[0], ybound[1]-ybound[0], fc='black',ec="black"))

p1.ax[0].plot(initial[0], initial[1], 'ko', markersize = 10)
p1.ax[0].plot(final[0], final[1], 'ko', markersize = 10)

p1.PartScreen(7.,6.)
p1.show()
