from tfc import utfc
from tfc.utils import TFCDictRobust, egrad, NllsClass, MakePlot

import numpy as onp
import jax.numpy as np
from jax import vmap, jacfwd, jit, lax

import tqdm
import pickle

from scipy.optimize import fsolve
from time import process_time as timer

## TEST PARAMETERS: ***************************************************
tol = np.finfo(float).eps
maxIter = 50

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
                  + np.dot(-xi['b']**2 *lamr(z,xi)[-1,:], xi['b']**2 * v(z,xi,IC)[-1,:])

L = jit(lambda xi,IC: np.hstack([Ls(xi,IC)[1:-1,:].flatten(), \
                                 L0(xi,IC).flatten(), \
                                 Ld0(xi,IC).flatten(), \
                                 Lf(xi,IC).flatten(), \
                                 Ldf(xi,IC).flatten(), \
                                 Htf(xi,IC)] ))


## INITIALIZE VARIABLES *************************************************************************************
xis   = onp.zeros((Hs(z).shape[1],3))
xic  = onp.zeros((Hc(z).shape[1],3))
b    = np.sqrt(2)*onp.ones(1)


xi = TFCDictRobust({'xis':xis,\
                    'xic':xic,\
                    'b':b})

IC = {'R0': np.zeros((3,)), 'V0': np.zeros((3,)), 'ag': np.zeros((3,))}

## NONLINEAR LEAST-SQUARES CLASS *****************************************************************************
nlls = NllsClass(xi,L,tol=tol,maxIter=maxIter,timer=True)

data = pickle.load(open('data/EOL_IC.pickle','rb'))
sol = {'loss': onp.zeros((data['R0'].shape[0])), 'it': onp.zeros((data['R0'].shape[0])), 'time': onp.zeros((data['R0'].shape[0]))}
## RUN TEST *************************************************************************************************
for i in tqdm.trange(data['R0'].shape[0]):
    R0 = data['R0'][i,:]
    V0 = data['V0'][i,:]

    ## scale initial conditons
    pscale = np.max(np.abs(R0))
    tscale = pscale/np.max(np.abs(V0))

    xi = TFCDictRobust({'xis':onp.zeros((Hs(z).shape[1],3)),\
                        'xic':onp.array([0.5 * (V0 - R0), -0.5*(V0 + R0)]),\
                        'b':np.sqrt(10.)*onp.ones(1)})


    IC['R0']    = R0 / pscale
    IC['V0']    = V0 * tscale/pscale
    IC['ag']    = np.array([0., 0., -1.62]) * tscale**2/pscale

    xi,it,time = nlls.run(xi,IC)

    sol['loss'][i] = np.max(np.abs(L(xi,IC)))
    sol['it'][i]   = it
    sol['time'][i] = time

## END: **************************************************************
# with open('data/EOL_Spec.pickle', 'wb') as handle:
#     pickle.dump(sol, handle)
