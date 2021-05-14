from tfc import utfc
from tfc.utils import TFCDictRobust, egrad, NllsClass, MakePlot

import numpy as onp
import jax.numpy as np
from jax import vmap, jacfwd, jit, lax

import tqdm
import pickle

from scipy.integrate import simps
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
                    + phi3(z)*(IC['V0']/xi['b']**2  - np.dot(pHs0,xi['xis'])) \
                    + phi4(z)*(                     - np.dot(pHsf,xi['xis']))

v = egrad(r)
a = egrad(v)


lam = lambda z, xi: np.dot(Hc(z),xi['xic'])


## FORM LOSS AND JACOBIAN ***********************************************************************************
Ls  = lambda xi,IC: xi['b']**4 * a(z,xi,IC) - IC['ag'] + lam(z,xi)
Htf = lambda xi,IC: np.dot(lam(z,xi)[-1,:],(-1./2.*lam(z,xi)[-1,:] + IC['ag']))
L = jit(lambda xi,IC: np.hstack([Ls(xi,IC).flatten(), Htf(xi,IC)] ))

## INITIALIZE VARIABLES *************************************************************************************
xis   = onp.zeros((Hs(z).shape[1],3))
xic   = onp.zeros((Hc(z).shape[1],3))
b = np.sqrt(2)*onp.ones(1)

xi = TFCDictRobust({'xis':xis,\
                    'xic':xic,\
                    'b':b})

IC = {'R0': np.zeros((3,)), 'V0': np.zeros((3,)), 'ag': np.zeros((3,))}

## NONLINEAR LEAST-SQUARES CLASS *****************************************************************************
nlls = NllsClass(xi,L,tol=tol,maxIter=maxIter,timer=True)

R0 = np.array([-150000., 30000., 15000.])
V0 = np.array([915., 0., 0.])

## scale initial conditons
pscale = np.max(np.abs(R0))
tscale = pscale/np.max(np.abs(V0))

IC['R0']    = R0 / pscale
IC['V0']    = V0 * tscale/pscale
IC['ag']    = np.array([0., 0., -1.62]) * tscale**2/pscale

xi,it,time = nlls.run(xi,IC)

## CONSTRUCT SOLUTION  **********************************************
t = (z-z[0])/xi['b']**2 * tscale

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

cost = 0.5 * simps(int,t)

##: print final answers to screen
print('\nFinal time [s]:\t' + str(t[-1]))
print('Cost:\t\t' + str(cost))
print('Comp time [ms]:\t' + str(time*1000))
print('Iterations:\t' + str(it))
print('Loss:\t\t' +  str(np.max(L(xi,IC))))

## Plot solution: **********************************************************************************

# range vs crosstrack
p1 = MakePlot('Range [km]','Crosstrack [km]')
p1.ax[0].plot(R[:,0]/1000.,R[:,1]/1000., 'k')
p1.ax[0].grid(True)
p1.PartScreen(8.,7.)
p1.show()

# 3d trajectoryAc
p2 = MakePlot(r'Range ($x$)',r'Crosstrack ($y$)', zlabs=r'Altitude ($z$)')
p2.ax[0].plot(R[:,0]/1000.,R[:,1]/1000.,R[:,2]/1000., 'k')
p2.ax[0].quiver(R[0::2,0]/1000.,R[0::2,1]/1000.,R[0::2,2]/1000., \
                Ac[0::2,0], Ac[0::2,1], Ac[0::2,2], \
                color='r', linewidth=1)
p2.ax[0].grid(True)

p2.ax[0].xaxis.labelpad = 10
p2.ax[0].yaxis.labelpad = 10
p2.ax[0].zaxis.labelpad = 10

p2.PartScreen(8.,7.)
p2.show()