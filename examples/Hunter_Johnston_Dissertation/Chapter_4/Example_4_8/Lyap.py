
import numpy as onp
import jax.numpy as np
from jax import vmap, jacfwd, jit, lax

from tfc import utfc
from tfc.utils import TFCDictRobust, egrad, NllsClass

from util import rich3_lyap, getL1L2, getJacobi

## Import loop and timing pacakges
import tqdm


## CONSTANTS: ***********************************************************************************************
m_E = 5.9724e24
m_M = 7.346e22
mu = m_M/(m_M + m_E)

## TEST PARAMETERS: *****************************************************************************************
tol = np.finfo(float).eps
maxIter = 20
basis = 'CP'
Lpt  = 'L1'

Cend = 2.91
nStep = 30

# number of points/basis functions
N = 140
m = 130

## GET CHEBYSHEV VALUES: ************************************************************************************

tfc  = utfc(N,4,m, basis=basis,      x0=-1.,xf=1.)

H = tfc.H
pH = tfc.dH

H0 = H(tfc.z[0:1])
Hf = H(tfc.z[-2:-1])

Hp0 = pH(tfc.z[0:1])
Hpf = pH(tfc.z[-2:-1])

## DEFINE THE ASSUMED SOLUTION: *****************************************************************************
z = tfc.z
z0 = z[0]
zf = z[-1]

R0 = lambda xi: np.array([xi['X'],xi['Y']]).flatten()
D0 = lambda xi: np.sqrt( (R0(xi)[0]+mu-1.)**2 + R0(xi)[1]**2 )
V0 = lambda xi: np.array([xi['dX'],xi['dY']]).flatten()/xi['b']**2/(1./D0(xi))

phi1 = lambda a:\
    np.expand_dims(1./(zf-z0)**3 * (-zf**2*(3.*z0-zf) + 6.*z0*zf*a - 3.*(z0+zf)*a**2 + 2.*a**3),1)
phi2 = lambda a:\
    np.expand_dims(1./(zf-z0)**3 * (-z0**2*(z0-3.*zf) - 6.*z0*zf*a + 3.*(z0+zf)*a**2 - 2.*a**3),1)
phi3 = lambda a:\
    np.expand_dims(1./(zf-z0)**2 * (-z0*zf**2 + zf*(2.*z0+zf)*a - (z0 + 2.*zf)*a**2 + a**3),1)
phi4 = lambda a:\
    np.expand_dims(1./(zf-z0)**2 * (-z0**2*zf + z0*(z0+2.*zf)*a - (2.*z0 + zf)*a**2 + a**3),1)

## CONSTRUCT THE CONSTRAINED EXPRESSION *********************************************************************
r = lambda z, xi: np.dot(H(z),xi['xis']) + phi1(z)*(R0(xi) - np.dot(H0, xi['xis'])) \
                                         + phi2(z)*(R0(xi) - np.dot(Hf, xi['xis'])) \
                                         + phi3(z)*(V0(xi) - np.dot(Hp0,xi['xis'])) \
                                         + phi4(z)*(V0(xi) - np.dot(Hpf,xi['xis']))

r1 = lambda z, xi: np.sqrt( (r(z,xi)[:,0]+mu   )**2 + r(z,xi)[:,1]**2 ) #+ r(z,xi)[:,2]**2)  # m1 to (x,y,z)
r2 = lambda z, xi: np.sqrt( (r(z,xi)[:,0]+mu-1.)**2 + r(z,xi)[:,1]**2 ) #+ r(z,xi)[:,2]**2)  # m2 to (x,y,z)

d2 = lambda z,xi: 1./r2(z,xi)

v = lambda z,xi: np.expand_dims(d2(z,xi),1) * xi['b']**2 * egrad(r)(z,xi)
a = lambda z,xi: np.expand_dims(d2(z,xi),1) * xi['b']**2 * egrad(v)(z,xi)

## LOSS FUNCTIONS AND JACOB *********************************************************************************

Jc = lambda Z, xi: (r(z,xi)[:,0]**2 + r(z,xi)[:,1]**2) \
                               + 2.*(1.-mu)/r1(z,xi) \
                               + 2.*mu/r2(z,xi) \
                               + (1.-mu)*mu \
                               - (v(z,xi)[:,0]**2 + v(z,xi)[:,1]**2)

Psi1 = lambda z, xi: np.hstack(( np.expand_dims( 2.*v(z,xi)[:,1] + r(z,xi)[:,0],1), \
                                 np.expand_dims(-2.*v(z,xi)[:,0] + r(z,xi)[:,1],1) ))

Psi2 = lambda z, xi: np.hstack(( np.expand_dims(r(z,xi)[:,0] + mu,1), \
                                 np.expand_dims(r(z,xi)[:,1],1) ))

Psi3 = lambda z, xi: np.hstack(( np.expand_dims(r(z,xi)[:,0] + mu - 1.,1), \
                                 np.expand_dims(r(z,xi)[:,1],1) ))


Ls = lambda z, xi: -a(z,xi) + Psi1(z,xi) \
                            - np.expand_dims((1.-mu)/r1(z,xi)**3,1) * Psi2(z,xi) \
                            - np.expand_dims(mu/r2(z,xi)**3,1)      * Psi3(z,xi)


## FORM LOSS AND JACOBIAN ***********************************************************************************
L = jit(lambda xi, C: np.hstack([Ls(z,xi).flatten(), Jc(z,xi)-C]))

## INITIALIZE VARIABLES *************************************************************************************
xis   = onp.zeros((H(z).shape[1],2))
X     = onp.ones(1)
Y     = onp.ones(1)
dX    = onp.ones(1)
dY    = onp.ones(1)
b     = onp.ones(1)

xi = TFCDictRobust({'xis':xis,\
                    'X':X,'Y':Y,\
                    'dX':dX,'dY':dY,\
                    'b':b})

# Creating dummy value for C so we can do NLLS
_, rvi, period, _ = rich3_lyap(mu, 0.05, 1)
C = np.array([getJacobi(rvi,mu)])

nlls = NllsClass(xi,L,C,tol=tol,maxIter=maxIter,timer=True)

## RUN TEST *************************************************************************************************
sol = { 'sol':onp.zeros((N,2,nStep)),\
        'res':onp.zeros((3*N,nStep)),\
        'time':onp.zeros((nStep)),\
        'iter':onp.zeros((nStep)),\
        'C':onp.zeros((nStep))}

## Initialization step
if Lpt == 'L1':
    _, rvi, period, _ = rich3_lyap(mu, 0.05, 1)
else:
    _, rvi, period, _ = rich3_lyap(mu, 0.05, 2)
C = np.array([getJacobi(rvi,mu)])

Cspan = np.linspace(C,Cend,nStep)

xi = TFCDictRobust({'xis':xis,\
                    'X':rvi[0:1,0],'Y':rvi[0:1,1],\
                    'dX':rvi[0:1,3],'dY':rvi[0:1,4],\
                    'b':onp.ones(1) * np.sqrt(2. / period)})


for i in tqdm.trange(nStep):
    xi,it,time = nlls.run(xi,C)

    sol['sol'][:,:,i] = r(z,xi)
    sol['res'][:,i]   = L(xi,C)
    sol['time'][i]    = time
    sol['iter'][i]    = it
    sol['C'][i]       = C

    print(np.max(L(xi,C)))
    ## Prepare for next Jacobi constant level
    C                      = Cspan[i+1]

## END ******************************************************************************************************
# import pickle
# with open('data/' + 'Lyap_' + str(basis) + '_' + str(Lpt) + '.pickle', 'wb') as handle:
#     pickle.dump(sol, handle)
