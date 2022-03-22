import numpy as onp
import jax.numpy as np
from jax import vmap, jacfwd, jit, lax

from tfc import utfc
from tfc.utils import  TFCDictRobust, egrad, NllsClass

from util import rich, getL1L2, getJacobi

## Import loop and timing pacakges
import tqdm

## CONSTANTS: ***********************************************************************************************
m_E = 5.9724e24
m_M = 7.346e22
mu = m_M/(m_M + m_E)

## TEST PARAMETERS: *****************************************************************************************
tol = np.finfo(float).eps
maxIter = 25
basis = 'CP'

# number of points/basis functions
N = 200
m = 190

hemi = 'N'
Lpt  = 'L2'

Az1 = 0.1
if Lpt == 'L1':
    _, rvi, period = rich(mu, Az1, 1, hemi, npts=1)
else:
    _, rvi, period = rich(mu, Az1, 2, hemi, npts=1)
Cstart = getJacobi(rvi,mu)

nTraj = 50
Cend = 2.95
Cspan = np.linspace(Cstart,Cend,nTraj)

# import pdb; pdb.set_trace()
## GET CHEBYSHEV VALUES: ************************************************************************************
tfc = utfc(N,4,m,basis=basis,x0=-1.,xf=1.)

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

R0 = lambda xi: np.array([xi['X'],xi['Y'],xi['Z']]).flatten()
V0 = lambda xi: np.array([xi['dX'],xi['dY'],xi['dZ']]).flatten()/xi['b']**2

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

r1 = lambda z, xi: np.sqrt( (r(z,xi)[:,0]+mu   )**2 + r(z,xi)[:,1]**2 + r(z,xi)[:,2]**2)  # m1 to (x,y,z)
r2 = lambda z, xi: np.sqrt( (r(z,xi)[:,0]+mu-1.)**2 + r(z,xi)[:,1]**2 + r(z,xi)[:,2]**2)  # m2 to (x,y,z)

v = lambda z,xi: xi['b']**2 * egrad(r)(z,xi)
a = lambda z,xi: xi['b']**2 * egrad(v)(z,xi)


## LOSS FUNCTIONS AND JACOB *********************************************************************************

Jc = lambda z, xi: (r(z,xi)[:,0]**2 + r(z,xi)[:,1]**2) \
                                    + 2.*(1.-mu)/r1(z,xi) \
                                    + 2.*mu/r2(z,xi) \
                                    + (1.-mu)*mu \
                                    - (v(z,xi)[:,0]**2 + v(z,xi)[:,1]**2 + v(z,xi)[:,2]**2)

Psi1 = lambda z, xi: np.hstack(( np.expand_dims( 2.*v(z,xi)[:,1] + r(z,xi)[:,0],1), \
                                 np.expand_dims(-2.*v(z,xi)[:,0] + r(z,xi)[:,1],1), \
                                 np.zeros((len(z),1)) ))

Psi2 = lambda z, xi: np.hstack(( np.expand_dims(r(z,xi)[:,0] + mu,1), \
                                 np.expand_dims(r(z,xi)[:,1],1), \
                                 np.expand_dims(r(z,xi)[:,2],1) ))

Psi3 = lambda z, xi: np.hstack(( np.expand_dims(r(z,xi)[:,0] + mu - 1.,1), \
                                 np.expand_dims(r(z,xi)[:,1],1), \
                                 np.expand_dims(r(z,xi)[:,2],1) ))


Ls = lambda z, xi: -a(z,xi) + Psi1(z,xi) \
                            - np.expand_dims((1.-mu)/r1(z,xi)**3,1) * Psi2(z,xi) \
                            - np.expand_dims(mu/r2(z,xi)**3,1)      * Psi3(z,xi)


## FORM LOSS AND JACOBIAN ***********************************************************************************
L = jit(lambda xi, C: np.hstack([Ls(z,xi).flatten(), Jc(z,xi)-C]) )

## INITIALIZE VARIABLES *************************************************************************************
xis   = onp.zeros((H(z).shape[1],3))
X     = rvi[0:1,0]
Y     = rvi[0:1,1]
Z     = rvi[0:1,2]
dX    = rvi[0:1,3]
dY    = rvi[0:1,4]
dZ    = rvi[0:1,5]
b     = onp.ones(1) * np.sqrt(2. / period)

xi = TFCDictRobust({'xis':xis,\
                    'X':X,'Y':Y,'Z':Z,\
                    'dX':dX,'dY':dY,'dZ':dZ,\
                    'b':b})

# Using dummy value of C for now
nlls = NllsClass(xi,L,1,tol=tol,maxIter=maxIter,timer=True)

## RUN TEST *************************************************************************************************
sol = {Lpt:{
        'sol':onp.zeros((N,3,len(Cspan))),\
        'res':onp.zeros((4*N,len(Cspan))),\
        'time':onp.zeros((len(Cspan))),\
        'iter':onp.zeros((len(Cspan))),\
        'C':onp.zeros((len(Cspan)))}}

for i in tqdm.trange(len(Cspan)):
    C = Cspan[i]
    xi,it,time = nlls.run(xi,C)

    print( str(C) + '  ' + str(np.max(L(xi,C))) )
    sol[Lpt]['sol'][:,:,i] = r(z,xi)
    sol[Lpt]['res'][:,i]   = L(xi,C)
    sol[Lpt]['time'][i]    = time
    sol[Lpt]['iter'][i]    = it
    sol[Lpt]['C'][i]       = C


## END ******************************************************************************************************
# import pickle
# with open('data/Halo' + '_' + str(basis) + '_' + str(Lpt) +  '_' + str(hemi) + '.pickle', 'wb') as handle:
#     pickle.dump(sol, handle)
