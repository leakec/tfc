# This is a function that solves the linear-nonlinear differential
# equation sequence with a shooting method
# Hunter Johnston - Texas A&M University
# Updated: 15 Mar 2021
################################################################################
# Differential Equation
#   y'' + yy'^a = exp(pi/2) - exp(pi/2 - x)
#
#   subject to: y(0)  = 9/10 + 1/10 exp(pi/2) (5 - 2 exp(pi/2))
#               y(pi) = exp(-pi/2)
################################################################################
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from time import process_time as timer

## Boundaries: *****************************************************************
x0 = 0.
x1 = np.pi/2.
xf = np.pi
tol = 2.220446049250313e-14

## Initial Conditions: *********************************************************
y0  = 9./10. + 1./10. * np.exp(np.pi/2.) * (5. - 2. * np.exp(np.pi/2.))
yf  = np.exp(-np.pi/2.)

## Compute true solution: ******************************************************
def ytrue(a):
    val = np.zeros_like(a)
    for i in range(0,len(a)):
        if a[i] <= np.pi/2.:
            val[i] = - 1./5. * np.exp(np.pi - 2.*a[i]) \
                     + 1./2. * np.exp(np.pi/2. - a[i]) \
                     + (9.*np.cos(a[i]) + 7.*np.sin(a[i])) / 10.
        else:
            val[i] = np.exp(np.pi/2. - a[i])
    return val

## DE: *************************************************************************
def de(t,y):
    dy = np.zeros(2)
    if t <= x1:
        a = 0
    else:
        a = 1

    dy[0]  = y[1]
    dy[1]  = -y[0]*y[1]**a - np.exp(np.pi-2.*t) + np.exp(np.pi/2-t)
    return dy

## Condition: ******************************************************************
def func(yp0):
    sol = solve_ivp(de, [0., xf], [y0, yp0[0]], method='RK45', rtol=tol, atol=tol)
    return sol.y[0,-1] - yf


## Fsolve: *********************************************************************
yp0 = 7.5

startTime = timer()
yp0 = fsolve(func, yp0, xtol=tol)

sol = solve_ivp(de, [0., xf], [y0, yp0[0]], method='RK45', rtol=tol, atol=tol)
time = timer() - startTime

x = sol.t
y = sol.y[0,:]
yp = sol.y[1,:]

err = np.abs(y - ytrue(x))

print()
print('Max Error: '             + str(np.max(err)))
print('yf Error: '             + str(np.abs(yf - y[-1])))
print('Computation time [ms]: ' + str(time*1000))
print()
