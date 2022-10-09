import numpy as onp
import jax.numpy as np
from jax import vmap, jacfwd, jit, lax

from tfc import utfc
from tfc.utils import TFCDictRobust, egrad

from util import rich3_lyap, getL1L2, getJacobi

from time import process_time as timer

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
Hf = H(tfc.z[-1:])

Hp0 = pH(tfc.z[0:1])
Hpf = pH(tfc.z[-1:])

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

def Jdark(xi,C):
     jacob = jacfwd(L,0)(xi,C)
     return np.hstack((jacob[k].reshape(jacob[k].shape[0],onp.prod(onp.array(xi[k].shape))) for k in xi.keys() ))
J = jit(lambda xi,C: Jdark(xi,C))

LS = jit(lambda Jacob,Loss: -np.dot( np.linalg.pinv(Jacob), Loss) )

## INITIALIZE VARIABLES *************************************************************************************
xis   = np.zeros((H(z).shape[1],2))
X     = np.ones(1)
Y     = np.ones(1)
dX    = np.ones(1)
dY    = np.ones(1)
b     = np.ones(1)

xi = TFCDictRobust({'xis':xis,\
                 'X':X,'Y':Y,\
                 'dX':dX,'dY':dY,\
                 'b':b})

 # ## JIT FUNCTIONS ******************************************************************************************
DXI = np.ones_like(xi.toArray())
val = {'xi':xi,'dxi':DXI,'it':0,'C':3.18,'tJac':0.,'tLS':0.}

## RUN TEST *************************************************************************************************
sol = { 'tLoss':onp.zeros((nStep)),\
        'tJac':onp.zeros((nStep)),\
        'tLS':onp.zeros((nStep)),\
        'C':onp.zeros((nStep))}

## Initialization step
if Lpt == 'L1':
    _, rvi, period, _ = rich3_lyap(mu, 0.05, 1)
else:
    _, rvi, period, _ = rich3_lyap(mu, 0.05, 2)
C = np.array([getJacobi(rvi,mu)])

Cspan = np.linspace(C,Cend,nStep)

val['xi']['xis']   = onp.zeros((H(z).shape[1],2))
val['xi']['X']     = rvi[0:1,0]
val['xi']['Y']     = rvi[0:1,1]
val['xi']['dX']    = rvi[0:1,3]
val['xi']['dY']    = rvi[0:1,4]
val['xi']['b']     = onp.ones(1) * np.sqrt(2. / period)


for i in range(nStep):
    ## Prepare for Jacobi constant level
    val['C']                      = Cspan[i]
    val['dxi']                    = DXI
    val['it']                     = 0
    val['tLoss']                  = 0.
    val['tJac']                   = 0.
    val['tLS']                    = 0.

    if i == 0:
        Los = L(val['xi'],val['C'])
        Jac = J(val['xi'],val['C'])
        LS(Jac,Los)

    while np.max(np.abs(L(val['xi'],val['C']))) > tol and val['it'] < maxIter and np.max(np.abs(val['dxi'])) > tol:
        start = timer()
        Loss = L(val['xi'],val['C']).block_until_ready()
        val['tLoss'] += timer() - start

        start = timer()
        Jacob = J(val['xi'],val['C']).block_until_ready()
        val['tJac'] += timer() - start

        start = timer()
        val['dxi'] = LS(Jacob,Loss).block_until_ready()
        val['tLS'] += timer() - start

        val['xi'] += val['dxi']
        val['it'] += 1

    sol['tLoss'][i]    = val['tLoss']
    sol['tJac'][i]    = val['tJac']
    sol['tLS'][i]     = val['tLS']
    sol['C'][i]          = val['C']

## END ******************************************************************************************************
# import pickle
# with open('data/' + 'timingLyap_' + str(basis) + '_' + str(Lpt) + '.pickle', 'wb') as handle:
#     pickle.dump(sol, handle)
