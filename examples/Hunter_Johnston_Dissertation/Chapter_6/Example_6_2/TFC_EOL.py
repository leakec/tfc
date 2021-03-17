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
nCx = 4
nCy = 0

## GET CHEBYSHEV VALUES **********************************************
stfc = utfc(N,nCx,ms,basis='CP',x0 = -1, xf = 1.)
ctfc = utfc(N,nCy,mc,basis='CP',x0 = -1, xf = 1.)

Hs  = stfc.H
pHs  = stfc.dH

Hs0 = Hs(stfc.z[0])
Hsf = Hs(stfc.z[-1])

pHs0 = pHs(stfc.z[0])
pHsf = pHs(stfc.z[-1])

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
Htf = lambda xi,IC: np.dot(lam(z,xi)[-1,:],(-1./2.*lam(z,xi)[-1,:] + IC['ag']))
L = jit(lambda xi,IC: Ls(xi,IC).flatten())

## INITIALIZE VARIABLES *************************************************************************************
xi = TFCDictRobust({'xis':onp.zeros((Hs(z).shape[1],3)),\
                    'xic':onp.zeros((Hc(z).shape[1],3))})

IC = {'R0': np.zeros((3,)), \
      'V0': np.zeros((3,)), \
      'ag': np.zeros((3,)), \
      'c':np.sqrt(2)*onp.ones(1)}

## NONLINEAR LEAST-SQUARES CLASS *****************************************************************************
nlls = NllsClass(xi,L,tol=tol,maxIter=2,timer=True)

data = pickle.load(open('data/EOL_IC.pickle','rb'))
sol = {'loss': onp.zeros((data['R0'].shape[0])), 'it': onp.zeros((data['R0'].shape[0])), 'time': onp.zeros((data['R0'].shape[0]))}

## INNERLOOP *************************************************************************************************
def InnerLoop(tf, xi, IC):
    global TIME, ITER
    IC['c'] = 2./tf

    xi,_,time = nlls.run(xi,IC)
    TIME += time
    ITER += 1

    loss1 = np.abs(Htf(xi,IC))
    loss2 = np.max(np.abs(L(xi,IC)))
    loss = np.max( np.hstack((loss1, loss2)) )
    return loss


## RUN TEST *************************************************************************************************
for i in tqdm.trange(data['R0'].shape[0]):
    R0 = data['R0'][i,:]
    V0 = data['V0'][i,:]

    ## scale initial conditons
    pscale = np.max(np.abs(R0))
    tscale = pscale/np.max(np.abs(V0))

    xi = TFCDictRobust({'xis':onp.zeros((Hs(z).shape[1],3)),\
                        'xic':onp.array([0.5 * (V0 - R0), -0.5*(V0 + R0)])})

    IC['R0']    = R0 / pscale
    IC['V0']    = V0 * tscale/pscale
    IC['ag']    = np.array([0., 0., -1.62]) * tscale**2/pscale

    tf_0 = 1.
    global TIME, IT
    TIME = 0.
    ITER = 0
    tf = fsolve(InnerLoop, tf_0, args=(xi, IC), xtol=1e-14, epsfcn=tol)

    ## solve with known tf
    IC['c'] = 2./tf
    xi,it,time = nlls.run(xi,IC)

    import pdb; pdb.set_trace()

    sol['loss'][i] = np.max(np.abs(L(xi,IC)))
    sol['it'][i]   = ITER
    sol['time'][i] = TIME

## END: **************************************************************
# with open('data/EOL_TFC.pickle', 'wb') as handle:
#     pickle.dump(sol, handle)
