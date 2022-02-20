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

def runLaneEmden(N, m, basis, k, xf):
## user defined parameters: ************************************************************************
# N      - number of discretization points
# m      - number of basis function terms
# basis  - basis function type
# k      - specific problem type, k >=0 (analytical solution known for k = 0, 1, and 5)

    ## problem initial conditions: *****************************************************************
    xspan = [0., xf] # problem domain range [x0, xf], where x₀ > 0
    y0  = 1.  # y(x0)  = 1
    y0p = 0.  # y'(x0) = 0
    nC  = 2   # number of constraints

    ## construct univariate tfc class: *************************************************************
    tfc = utfc(N, nC, int(m), basis = basis, x0=xspan[0], xf=xspan[1])
    x = tfc.x

    H = tfc.H
    dH = tfc.dH
    H0 = H(x[0:1])
    H0p = dH(x[0:1])

    ## define tfc constrained expression and derivatives: ******************************************
    # switching function
    phi1 = lambda x: np.ones_like(x)
    phi2 = lambda x: x

    # tfc constrained expression
    y = lambda x,xi: np.dot(H(x),xi) + phi1(x)*(y0  - np.dot(H0,xi)) + phi2(x)*(y0p - np.dot(H0p,xi))
    yp = egrad(y)
    ypp = egrad(yp)

    ## define the loss function: *******************************************************************
    L = lambda xi: x*ypp(x,xi) + 2.*yp(x,xi) + x*y(x,xi)**k

    ## solve the problem via nonlinear least-squares ***********************************************
    xi = np.zeros(H(x).shape[1])

    # if k==0 or k==1, the problem is linear
    if k == 0 or k == 1:
        xi,time = LS(xi,L,timer=True)
        iter = 1

    else:
        xi,iter,time = NLLS(xi,L,timer=True)

    ## compute the error (if k = 0, 1, or 5): ******************************************************
    if k == 0:
        ytrue = 1. - 1./6. * x**2
    elif k == 1:
        ytrue = onp.ones_like(x)
        ytrue[1:] = np.sin(x[1:]) / x[1:]
    elif k == 5:
        ytrue = (1. + x**2/3)**(-1/2)
    else:
        ytrue = np.empty_like(x)

    err = np.abs(y(x,xi) - ytrue)

    ## compute the residual of the loss vector: ****************************************************
    res = np.abs(L(xi))
    
    return x, y(x,xi), err, res

####################################################################################################
####################################################################################################
## run the lane-emden for each case of k
# runLaneEmden(N, m, basis, k, x)
x_k0, sol_k0, err_k0, res_k0 = runLaneEmden(100, 2,  'CP', 0, 10.)
x_k1, sol_k1, err_k1, res_k1 = runLaneEmden(100, 22, 'CP', 1, 10.)
x_k5, sol_k5, err_k5, res_k5 = runLaneEmden(100, 62, 'CP', 5, 10.)
####################################################################################################
####################################################################################################

## plotting: ***************************************************************************************

# figure 1: solution
p1 = MakePlot(r'$x$',r'$y(x)$')
p1.ax[0].plot(x_k0,sol_k0, label='k=0')
p1.ax[0].plot(x_k1,sol_k1, label='k=1')
p1.ax[0].plot(x_k5,sol_k5, label='k=5')
p1.ax[0].grid(True)
p1.ax[0].set_xlim(0., 10.)
p1.ax[0].set_ylim(-0.5, 1)
p1.ax[0].legend()
p1.PartScreen(7.,6.)
p1.show()

# figure 2: residual
p2 = MakePlot(r'$x$',r'$|L(\xi)|$')
p2.ax[0].plot(x_k0,res_k0, '*', label='k=0')
p2.ax[0].plot(x_k1,res_k1, '*', label='k=1')
p2.ax[0].plot(x_k5,res_k5, '*', label='k=5')
p2.ax[0].grid(True)
p2.ax[0].set_yscale('log')
p2.ax[0].set_xlim(0., 10.)
p2.ax[0].legend()
p2.PartScreen(7.,6.)
p2.show()

# figure 3: error (if k = 0, 1, or 5)
p3 = MakePlot(r'$x$',r'$|y_{true} - y(x)|$')
p3.ax[0].plot(x_k0,err_k0, '*', label='k=0')
p3.ax[0].plot(x_k1,err_k1, '*', label='k=1')
p3.ax[0].plot(x_k5,err_k5, '*', label='k=5')
p3.ax[0].grid(True)
p3.ax[0].set_yscale('log')
p3.ax[0].set_xlim(0., 10.)
p3.ax[0].legend()
p3.PartScreen(7.,6.)
p3.show()
