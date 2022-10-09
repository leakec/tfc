# This is a function that solves the convection diffusion equation with TFC
# This script breaks the problem into two segments. 
# The segment lengths are chosen using nonlinear least-squares
# Hunter Johnston - Texas A&M University
# Updated: 15 Mar 2021
################################################################################
# Differential Equation
#   y'' - Pe y = 0
#
#   subject to: y(0)  = 9/10 + 1/10 exp(pi/2) (5 - 2 exp(pi/2))
#               y(pi) = exp(-pi/2)
################################################################################
from tfc import utfc
from tfc.utils import MakePlot, TFCDict, egrad, NLLS, step
import jax.numpy as np
from jax import jit

import numpy as onp
import tqdm
from time import process_time as timer
## Analytical solution: ********************************************************
soln = lambda x: (1.-np.exp(Pe*(x-1.)))/(1.-np.exp(-Pe))
dsoln = lambda x: egrad(soln,0)(x)
ddsoln = lambda x: egrad(dsoln,0)(x)

# Constants used in the differential equation:
Pe = 10**6
tol = 1e-13

xI = 0.
xf = 1.
yi = 1.
yf = 0.

xpBound = 1.-1e-3

# Create the ToC Class:
N = 200
m = 190
nC = 3
tfc = utfc(N,nC,m,basis='CP',x0=-1,xf=1)

# Get the Chebyshev polynomials
H = tfc.H
dH = tfc.dH
H0 = H(tfc.z[0:1])
Hf = H(tfc.z[-1:])
Hd0 = dH(tfc.z[0:1])
Hdf = dH(tfc.z[-1:],)

# Create the constraint expression and its derivatives
z = tfc.z

xpdark = lambda xi: 2./xi['b']**2 + xI

xp = lambda xi: xpdark(xi)+(xpBound-xpdark(xi))*step(xpdark(xi)-xpBound)

c1 = lambda xi: 2./(xp(xi)-xI)
c2 = lambda xi: 2./(xf-xp(xi))

x1 = lambda z,xi: (z+1.)/c1(xi)+xI
x2 = lambda z,xi: (z+1.)/c2(xi)+xp(xi)

phi1_s1 = lambda a: (1.-2.*a+a**2)/4.
phi2_s1 = lambda a: (3.+2.*a-a**2)/4.
phi3_s1 = lambda a: (-1.+a**2)/2.

phi1_s2 = lambda a: (3.-2.*a-a**2)/4.
phi2_s2 = lambda a: (1.-a**2)/2.
phi3_s2 = lambda a: (1.+2.*a+a**2)/4.

y1 = lambda z,xi: \
np.dot(H(z),xi['xi1'])+phi1_s1(z)*(yi                     -np.dot(H0, xi['xi1']))\
                      +phi2_s1(z)*(xi['y']                -np.dot(Hf, xi['xi1']))\
                      +phi3_s1(z)*(xi['yd']/c1(xi)        -np.dot(Hdf,xi['xi1']))
ydz1 = egrad(y1,0)
yddz1 = egrad(ydz1,0)


yd1 = lambda z,xi: ydz1(z,xi)*c1(xi)
ydd1 = lambda z,xi: yddz1(z,xi)*c1(xi)**2

y2 = lambda z,xi: \
np.dot(H(z),xi['xi2'])+phi1_s2(z)*(xi['y']               -np.dot(H0, xi['xi2']))\
                      +phi2_s2(z)*(xi['yd']/c2(xi)       -np.dot(Hd0,xi['xi2']))\
                      +phi3_s2(z)*(yf                    -np.dot(Hf, xi['xi2']))

ydz2 = egrad(y2,0)
yddz2 = egrad(ydz2,0)
yd2 = lambda z,xi: ydz2(z,xi)*c2(xi)
ydd2 = lambda z,xi: yddz2(z,xi)*c2(xi)**2

L1 = lambda z,xi: ydd1(z,xi)-Pe*yd1(z,xi)
L2 = lambda z,xi: ydd2(z,xi)-Pe*yd2(z,xi)

L = lambda xi: np.hstack(( L1(z,xi), L2(z,xi) ))


# Create the residual and jacobians
xi1  = onp.zeros(H(z).shape[1])
xi2  = onp.zeros(H(z).shape[1])
y    = onp.zeros(1)
yd   = onp.zeros(1)
b    = onp.ones(1) * np.sqrt(2./0.5)

xi = TFCDict({'xi1':xi1,'xi2':xi2,'y':y,'yd':yd,'b':b})

## SOLVE THE SYSTEM *************************************************
xi,it,time = NLLS(xi,L,timer=True)


X = np.hstack((x1(z,xi), x2(z,xi)))
Y = np.hstack((y1(z,xi), y2(z,xi) ))


# p1 = MakePlot(onp.array([['x (m)']]),onp.array([['y (m)']]))
# p1.ax[0].plot(X,Y,label='TFC Solution')
# p1.ax[0].plot(X,soln(X),label='Analytical Solution')
# p1.ax[0].legend()
# p1.show()


print('{:.2e} & {:.2e} & {:.5f} & {:.2f}'.format(np.max(np.abs(Y - soln(X))), np.max(np.abs(L(xi))), xp(xi)[0].tolist(), time ))
