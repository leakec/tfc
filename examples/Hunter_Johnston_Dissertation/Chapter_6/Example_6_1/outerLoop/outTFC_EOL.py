from tfc import utfc
from tfc.utils import TFCDictRobust, egrad, NllsClass, MakePlot

import numpy as onp
import jax.numpy as np
from jax import vmap, jacfwd, jit, lax

import tqdm
import pickle

from scipy.optimize import fsolve
from scipy.integrate import simps
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
nCx = 4
nCy = 0

## GET CHEBYSHEV VALUES **********************************************
stfc = utfc(N,nCx,ms,basis='CP',x0 = -1, xf = 1.)
ctfc = utfc(N,nCy,mc,basis='CP',x0 = -1, xf = 1.)

Hs  = stfc.H
pHs  = stfc.dH

Hs0 = Hs(stfc.z[0:1])
Hsf = Hs(stfc.z[-1:])

pHs0 = pHs(stfc.z[0:1])
pHsf = pHs(stfc.z[-1:])

Hc  = ctfc.H

## DEFINE THE ASSUMED SOLUTION **************************************
z = stfc.z
z0 = z[0]
zf = z[-1]


## DEFINE SWITCHING FUNCTIONS ***************************************
phi1 = lambda a: np.expand_dims(\
                 1./(zf-z0)**3 * (-zf**2*(3.*z0-zf) + 6.*z0*zf*a - 3.*(z0+zf)*a**2 + 2.*a**3),1)
phi2 = lambda a: np.expand_dims(\
                1./(zf-z0)**3 * (-z0**2*(z0-3.*zf) - 6.*z0*zf*a + 3.*(z0+zf)*a**2 - 2.*a**3),1)
phi3 = lambda a: np.expand_dims(\
                1./(zf-z0)**2 * (-z0*zf**2 + zf*(2.*z0+zf)*a - (z0+2.*zf)*a**2 + a**3),1)
phi4 = lambda a: np.expand_dims(\
                1./(zf-z0)**2 * (-z0**2*zf + z0*(z0+2.*zf)*a - (2.*z0+zf)*a**2 + a**3),1)


## DEFINE CONSTRAINED EXPRESSION *************************************
r = lambda z, xi, IC: np.dot(Hs(z),xi['xis']) \
                    + phi1(z)*(IC['R0']             - np.dot(Hs0, xi['xis'])) \
                    + phi2(z)*(                     - np.dot(Hsf, xi['xis'])) \
                    + phi3(z)*(IC['V0']/IC['c']     - np.dot(pHs0,xi['xis'])) \
                    + phi4(z)*(                     - np.dot(pHsf,xi['xis']))

v = egrad(r)
a = egrad(v)


lam = lambda z, xi: np.dot(Hc(z),xi['xic'])


## FORM LOSS AND JACOBIAN ***********************************************************************************
Ls  = lambda xi,IC: IC['c']**2 * a(z,xi,IC) - IC['ag'] + lam(z,xi)
Htf = lambda xi,IC: np.dot(lam(z,xi)[-1,:],(-1./2.*lam(z,xi)[-1,:] + IC['ag'])) + IC['Gam']
L = jit(lambda xi,IC: Ls(xi,IC).flatten())

## INITIALIZE VARIABLES *************************************************************************************
xis   = onp.zeros((Hs(z).shape[1],3))
xic   = onp.zeros((Hc(z).shape[1],3))

if W == False:
    c = 2.*onp.ones(1)
else:
    c = 10.*onp.ones(1)


xi = TFCDictRobust({'xis':xis,\
                    'xic':xic})

IC = {'R0': np.zeros((3,)), \
      'V0': np.zeros((3,)), \
      'ag': np.zeros((3,)), \
      'Gam': np.zeros((1,)), \
      'c': 2.*onp.ones(1)}

## NONLINEAR LEAST-SQUARES CLASS *****************************************************************************
nlls = NllsClass(xi,L,IC,maxIter=2,timer=True)

R0 = np.array([500000., 100000., 50000.])
V0 = np.array([-3000., 0., 0.])

## scale initial conditons
pscale = np.max(np.abs(R0))
tscale = pscale/np.max(np.abs(V0))

IC['R0']    = R0 / pscale
IC['V0']    = V0 * tscale/pscale
IC['ag']    = np.array([0., 0., -5.314961]) * tscale**2/pscale
IC['Gam']   = Gam * tscale**4/pscale**2

global it
it = 0
def Innerloop(tf,xi,IC):
    global it

    IC['c'] = 2./tf

    it += 1
    xi,_,time = nlls.run(xi,IC)
    loss1 = np.max(np.abs(L(xi,IC)))
    loss2 = np.max(np.abs(Htf(xi,IC)))
    return np.max(np.hstack((loss1,loss2)))


t0 = 2./IC['c']

start = timer()
tf = fsolve(Innerloop, t0, args=(xi,IC), xtol=1e-13,epsfcn=tol)
time = timer() - start

IC['c'] = 2./tf
xi,_,_ = nlls.run(xi,IC)

## CONSTRUCT SOLUTION  **********************************************
t = (z-z[0])/IC['c'] * tscale
IC['Gam']= IC['Gam'] * pscale**2/tscale**4

R = r(z,xi,IC) * pscale
V = v(z,xi,IC) * pscale/tscale

LamV = lam(z,xi) * pscale/tscale**2
LamR = -IC['c'] * egrad(lam)(z,xi) * pscale/tscale**3

Ac = - LamV

Ham = onp.zeros(len(t))
int = onp.zeros(len(t))
a_mag = onp.zeros(len(t))
for i in range(0,len(t)):
    int[i] = np.dot(Ac[i,:],Ac[i,:])
    Ham[i] = 0.5*int[i] + np.dot(LamR[i,:],V[i,:]) + np.dot(LamV[i,:],IC['ag'] + Ac[i,:])
    a_mag[i] = np.linalg.norm(Ac[i,:])

cost = IC['Gam']* t[-1] +  0.5 * simps(int,t)

loss1 = np.max(np.abs(L(xi,IC)))
loss2 = np.max(np.abs(Htf(xi,IC)))

##: print final answers to screen
print('\nFinal time [s]:\t' + str(t[-1]))
print('Cost:\t\t' + str(cost))
print('Comp time [ms]:\t' + str(time*1000))
print('Iterations:\t' + str(it))
print('Loss:\t\t' +  str(np.max(np.hstack((loss1,loss2)))))
