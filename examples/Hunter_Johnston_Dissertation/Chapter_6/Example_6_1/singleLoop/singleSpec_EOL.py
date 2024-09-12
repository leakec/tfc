from tfc import utfc
from tfc.utils import TFCDictRobust, egrad, NllsClass, MakePlot

import numpy as onp
import jax.numpy as np
from jax import vmap, jacfwd, jit, lax

import tqdm
import pickle

from scipy.integrate import simpson
from time import process_time as timer

## TEST PARAMETERS: ***************************************************
tol = np.finfo(float).eps
maxIter = 50

W = False

if W == False:
    Gam = 0.
else:
    Gam = 100.

## CONSTANTS: *********************************************************
# Number of points to use
N = 100

# Number of basis functions to use
ms = 30
mc = 1

# Number of constraints
nCx = 0
nCy = 0

## GET CHEBYSHEV VALUES **********************************************
stfc = utfc(N,nCx,ms,basis='CP',x0 = -1, xf = 1.)
ctfc = utfc(N,nCy,mc,basis='CP',x0 = -1, xf = 1.)

Hs  = stfc.H
Hc  = ctfc.H

## DEFINE THE ASSUMED SOLUTION **************************************
z = stfc.z
z0 = z[0]
zf = z[-1]


## DEFINE CONSTRAINED EXPRESSION *************************************
r = lambda z, xi, IC: np.dot(Hs(z),xi['xis'])
v = egrad(r,0)
a = egrad(v,0)

lam = lambda z, xi: np.dot(Hc(z),xi['xic'])
lamr = egrad(lam,0)


## FORM LOSS AND JACOBIAN ***********************************************************************************
L0  = lambda xi,IC: r(z,xi,IC)[0,:] - IC['R0']
Ld0 = lambda xi,IC: xi['b']**2 * v(z,xi,IC)[0,:] - IC['V0']
Lf  = lambda xi,IC: r(z,xi,IC)[-1,:]
Ldf = lambda xi,IC: xi['b']**2 * v(z,xi,IC)[-1,:]

Ls  = lambda xi,IC: xi['b']**4 * a(z,xi,IC) - IC['ag'] + lam(z,xi)


# Htf = lambda xi,IC: np.dot(lam(z,xi)[-1,:],(-1./2.*lam(z,xi)[-1,:] + IC['ag']))
# Updated because need to at lam_r * v term for spectral method
Htf = lambda xi,IC: np.dot(lam(z,xi)[-1,:],(-1./2.*lam(z,xi)[-1,:] + IC['ag'])) \
                  + np.dot(-xi['b']**2 *lamr(z,xi)[-1,:], xi['b']**2 * v(z,xi,IC)[-1,:]) + IC['Gam']

L = jit(lambda xi,IC: np.hstack([Ls(xi,IC)[1:-1,:].flatten(), \
                                 L0(xi,IC).flatten(), \
                                 Ld0(xi,IC).flatten(), \
                                 Lf(xi,IC).flatten(), \
                                 Ldf(xi,IC).flatten(), \
                                 Htf(xi,IC)] ))


## INITIALIZE VARIABLES *************************************************************************************
xis  = onp.zeros((Hs(z).shape[1],3))
xic  = onp.zeros((Hc(z).shape[1],3))

if W == False:
    b = np.sqrt(2)*onp.ones(1)
else:
    b = np.sqrt(10)*onp.ones(1)


xi = TFCDictRobust({'xis':xis,\
                    'xic':xic,\
                    'b':b})

IC = {'R0': np.zeros((3,)), 'V0': np.zeros((3,)), 'ag': np.zeros((3,)), 'Gam': np.zeros((1,))}

## NONLINEAR LEAST-SQUARES CLASS *****************************************************************************
nlls = NllsClass(xi,L,IC,tol=tol,maxIter=maxIter,timer=True)

R0 = np.array([500000., 100000., 50000.])
V0 = np.array([-3000., 0., 0.])

## scale initial conditons
pscale = np.max(np.abs(R0))
tscale = pscale/np.max(np.abs(V0))

IC['R0']    = R0 / pscale
IC['V0']    = V0 * tscale/pscale
IC['ag']    = np.array([0., 0., -5.314961]) * tscale**2/pscale
IC['Gam']   = Gam * tscale**4/pscale**2

xi,it,time = nlls.run(xi,IC)

## CONSTRUCT SOLUTION  **********************************************
t = (z-z[0])/xi['b']**2 * tscale
IC['Gam']= IC['Gam'] * pscale**2/tscale**4

R = r(z,xi,IC) * pscale
V = v(z,xi,IC) * pscale/tscale

LamV = lam(z,xi) * pscale/tscale**2
LamR = -xi['b']**2 * egrad(lam)(z,xi) * pscale/tscale**3

Ac = - LamV

Ham = onp.zeros(len(t))
int = onp.zeros(len(t))
a_mag = onp.zeros(len(t))
for i in range(0,len(t)):
    int[i] = np.dot(Ac[i,:],Ac[i,:])
    Ham[i] = 0.5*int[i] + np.dot(LamR[i,:],V[i,:]) + np.dot(LamV[i,:],IC['ag'] + Ac[i,:])
    a_mag[i] = np.linalg.norm(Ac[i,:])

cost = IC['Gam']* t[-1] +  0.5 * simpson(int,x=t)


##: print final answers to screen
print('\nFinal time [s]:\t' + str(t[-1]))
print('Cost:\t\t' + str(cost))
print('Comp time [ms]:\t' + str(time*1000))
print('Iterations:\t' + str(it))
print('Loss:\t\t' +  str(np.max(L(xi,IC))))
