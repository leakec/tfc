# This is a function that solves the BVP using TFC
# Hunter Johnston - Texas A&M University
# Updated: 15 Mar 2021
#####################################################################
# Differential Equation
#   y'' + yy' = exp(-2x) sin(x) [cos(x) -sin(x)] - 2exp(-x)cos(x)
#
#   subject to: y(0) = 0, y(pi) = 0
#####################################################################
import numpy as np
from scipy.integrate import solve_bvp

from time import process_time as timer
#####################################################################

def BVP_ode(solver, tol):
    xspan = [0., np.pi]

    N = 100
    k = np.arange(0, N, 1, dtype=int)
    z = - np.cos( (k * np.pi) / (N - 1) )
    c = (z[-1] - z[0]) / (xspan[1] - xspan[0])
    x = xspan[0] + 1./c * (z - z[0])

    def bc(ya,yb):
        return np.array([ya[0], yb[0]])

    def dydx(x, y):
        dy1 = y[1]
        f = np.exp(-2.*x) * np.sin(x) * (np.cos(x) - np.sin(x)) - 2.*np.exp(-x)*np.cos(x)
        dy2 = - y[0]*dy1 + f

        return np.array([dy1, dy2])


    y = np.zeros((2, x.size))

    start = timer()
    sol = solve_bvp(dydx, bc, x, y, tol=tol, bc_tol = tol)
    end = timer()

    X = sol.x
    Y = sol.y[0,:]


    ## COMPUTE ERROR AND RESIDUAL ***************************************
    ## Compute true solution
    ytrue = lambda x: np.exp(-x) * np.sin(x)

    err = np.linalg.norm(Y - ytrue(X))
    time = end - start

    return err, time
