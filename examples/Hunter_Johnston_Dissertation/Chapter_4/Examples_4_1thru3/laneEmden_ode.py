# This is a function that solves the general Lane-Emden equation using TFC
# Hunter Johnston - Texas A&M University
# Updated: 10 Mar 2021
##################################################################
# Differential Equation
#   y'' + 2/x y' + y^k = 0
#
#   subject to: y(0) = 1, y'(0) = 0
##################################################################
import numpy as np
from scipy import integrate
from time import process_time as timer
##################################################################
def laneEmden_ode(solver, xspan, type, tol):
    N = 100
    k = np.arange(0, N, 1, dtype=int)
    z = - np.cos( (k * np.pi) / (N - 1) )
    c = (z[-1] - z[0]) / (xspan[1] - xspan[0])
    x = xspan[0] + 1./c * (z - z[0])

    def dydx(x, y):
        dy1 = y[1]

        if x == 0:
            dy2 = 0.
        else:
            dy2 = -2./x * dy1 - y[0]**type

        return np.array([dy1, dy2])


    y0 = [1., 0.]

    start = timer()
    sol = integrate.solve_ivp(dydx, [xspan[0], xspan[-1]], y0, t_eval = x, method=solver, rtol=tol, atol=tol)
    end = timer()

    X = sol.t
    Y = sol.y[0,:]


    ## COMPUTE ERROR AND RESIDUAL ***************************************
    ## Compute true solution
    if type == 0:
        def ytrue(x):
            val = np.zeros_like(x)
            val[0] = 1.
            val[1:] = 1. - 1./6. * x[1:]**2
            return val
    elif type == 1:
        def ytrue(x):
            val = np.zeros_like(x)
            val[0] = 1.
            val[1:] = np.sin(x[1:]) / x[1:]
            return val

    elif type == 5:
        def ytrue(x):
            val = np.zeros_like(x)
            val[0] = 1.
            val[1:] = (1. + x[1:]**2/3)**(-1/2)
            return val
    else:
        def ytrue(x):
            return np.nan * np.ones_like(x)

    err = np.linalg.norm(Y - ytrue(X))
    time = end - start

    return err, time
