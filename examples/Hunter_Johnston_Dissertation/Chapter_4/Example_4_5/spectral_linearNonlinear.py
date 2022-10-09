# This is a function that solves the linear-nonlinear differential
# equation sequence with spectral method
# Hunter Johnston - Texas A&M University
# Updated: 15 Mar 2021
################################################################################
# Differential Equation
#   y'' + yy'^a = exp(pi/2) - exp(pi/2 - x)
#
#   subject to: y(0)  = 9/10 + 1/10 exp(pi/2) (5 - 2 exp(pi/2))
#               y(pi) = exp(-pi/2)
################################################################################
from tfc import utfc
from tfc.utils import TFCDict, egrad, NLLS
import jax.numpy as np
from jax import jit

import numpy as onp
import tqdm
################################################################################
N = 100
m = 20 + 3
basis = 'CP'
tol = 1e-16
iterMax = 50

## Boundaries: *****************************************************************
x0 = 0.
x1 = np.pi/2.
xf = np.pi

## Initial Conditions: *********************************************************
y0  = 9./10. + 1./10. * np.exp(np.pi/2.) * (5. - 2. * np.exp(np.pi/2.))
yf  = np.exp(-np.pi/2.)


nC  = 0 # number of constraints

## Compute true solution: ******************************************************
def ytrue(a):
    val = onp.zeros_like(a)
    for i in range(0,len(a)):
        if a[i] <= np.pi/2.:
            val[i] = - 1./5. * np.exp(np.pi - 2.*a[i]) \
                     + 1./2. * np.exp(np.pi/2. - a[i]) \
                     + (9.*np.cos(a[i]) + 7.*np.sin(a[i])) / 10.
        else:
            val[i] = np.exp(np.pi/2. - a[i])
    return val


## GET CHEBYSHEV VALUES: *******************************************************

# First segment
tfc1 = utfc(N,nC,m,basis = basis, x0 = x0, xf = x1)
xs1 = tfc1.x

Hs1 = tfc1.H
dHs1 = tfc1.dH

H0s1 = Hs1(tfc1.x[0:1])
Hfs1 = Hs1(tfc1.x[-1:])

Hfps1 = dHs1(tfc1.x[-1:])

# Second segment
tfc2 = utfc(N,nC,m,basis = basis, x0 = x1, xf = xf)
xs2 = tfc2.x

Hs2 = tfc2.H
dHs2 = tfc2.dH

H0s2 = Hs2(tfc2.x[0:1])
Hfs2 = Hs2(tfc2.x[-1:])

H0ps2 = dHs2(tfc2.x[0:1])


## DEFINE THE ASSUMED SOLUTION: ************************************************

# First segment
ys1 = lambda x, xi: np.dot(Hs1(x),xi['xi1'])
yps1  = egrad(ys1,0)
ypps1 = egrad(yps1,0)

# Second segment
ys2 = lambda x, xi: np.dot(Hs2(x),xi['xi2'])
yps2  = egrad(ys2,0)
ypps2 = egrad(yps2,0)

## DEFINE LOSS AND JACOB *******************************************************
f = lambda x: -np.exp(np.pi - 2.*x) + np.exp(np.pi/2. - x) 

L0  = lambda xi: ys1(xs1,xi)[0] - y0
L1  = lambda xi: (ypps1(xs1, xi) + ys1(xs1, xi) - f(xs1))[1:-1]
Li  = lambda xi: ys1(xs1,xi)[-1] - ys2(xs2,xi)[0]
Lip = lambda xi: yps1(xs1,xi)[-1] - yps2(xs2,xi)[0]
L2  = lambda xi: (ypps2(xs2, xi) + ys2(xs2, xi)*yps2(xs2, xi) - f(xs2))[1:-1]
Lf  = lambda xi: ys2(xs2,xi)[-1] - yf

L = jit( lambda xi: np.hstack(( L0(xi), L1(xi), Li(xi), Lip(xi), L2(xi), Lf(xi) )) )



## SOLVE THE SYSTEM ******************************************************************
xi1 =  onp.zeros(Hs1(xs1).shape[1])
xi2 =  onp.zeros(Hs2(xs2).shape[1])

xi0 = TFCDict({'xi1':xi1,'xi2':xi2})

xi,it,time = NLLS(xi0,L,timer=True)


## COMPUTE ERROR AND RESIDUAL ******************************************************************
x = np.hstack((xs1,xs2))

y  = np.hstack(( ys1(xs1,xi), ys2(xs2,xi) ))
yp = np.hstack(( yps1(xs1,xi), yps2(xs2,xi) ))

print()
print('Max Error: '             + str(np.max(np.abs(y - ytrue(x)))))
print('y0 Error: '              + str( np.abs(y[0] - y0) ) )
print('y1 Error: '              + str( np.abs(Li(xi)) ) )
print('y1_x Error: '            + str( np.abs(Lip(xi)) ) )
print('yf Error: '              + str( np.abs(y[-1] - yf) ) )
print('Max Loss: '              + str(np.max(np.abs(L(xi)))))
print('Computation time [ms]: ' + str(time*1000))
print()
