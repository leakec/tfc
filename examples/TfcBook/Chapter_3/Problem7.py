# This script solves Problem #7 of Chapter 3 in the TFC book
####################################################################################################
# Differential Equations
#   ẍ =  2ẏ + x - (x + μ)*(1 - μ)/(R₁)³ - (x + μ - 1)*μ/(R₂)³
#   ÿ = -2ẋ + y -  y     *(1 - μ)/(R₁)³ -  y         *μ/(R₂)³   
    #   C = x² + y² + 2(1 - μ )/R₁ + 2μ/R₂ + μ(1 - μ) - (ẋ + ẏ)

#   subject to: x(0) = x(T),    ẋ(0) = ẋ(T)
#               y(0) = y(T),    ẏ(0) = ẏ(T)
####################################################################################################
from tfc import utfc
from tfc.utils import TFCDictRobust, egrad, NLLS, MakePlot
import jax.numpy as np
import numpy as onp

# might be able to remove
# from jax import jit
from jax import vmap, jacfwd, jit, lax


#need to add this code in main file
####################################################################################################

## constants: **************************************************************************************
mu = 0.01215
C  = 3.04856909

## initial guess values: ***************************************************************************
r_init = [0.79, 0.]     # [7.91965585e-01 3.73375471e-11] 
v_init = [0.,   0.42]   # [3.57685915e-11 4.02225959e-01]
T_init = 2.7            # 2.7034019846274746


## user defined parameters: ************************************************************************
N       = 140   # number of discretization points
m       = 130   # number of basis function terms
basis   = 'CP'  # basis function type

## construct univariate tfc class: *****************************************************************
tfc  = utfc(N, 4, m, basis=basis, x0=-1., xf=1.)

H = tfc.H
dH = tfc.dH

H0 = H(tfc.z[0])
Hf = H(tfc.z[-1])

Hp0 = dH(tfc.z[0])
Hpf = dH(tfc.z[-1])

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

R1 = lambda z, xi: np.sqrt( (r(z,xi)[:,0]+mu   )**2 + r(z,xi)[:,1]**2 ) 
R2 = lambda z, xi: np.sqrt( (r(z,xi)[:,0]+mu-1.)**2 + r(z,xi)[:,1]**2 )

d2 = lambda z,xi: 1./R2(z,xi)

v = lambda z,xi: np.expand_dims(d2(z,xi),1) * xi['b']**2 * egrad(r)(z,xi)
a = lambda z,xi: np.expand_dims(d2(z,xi),1) * xi['b']**2 * egrad(v)(z,xi)

## LOSS FUNCTIONS AND JACOB *********************************************************************************

Jc = lambda Z, xi: (r(z,xi)[:,0]**2 + r(z,xi)[:,1]**2) \
                               + 2.*(1.-mu)/R1(z,xi) \
                               + 2.*mu/R2(z,xi) \
                               + (1.-mu)*mu \
                               - (v(z,xi)[:,0]**2 + v(z,xi)[:,1]**2)

Psi1 = lambda z, xi: np.hstack(( np.expand_dims( 2.*v(z,xi)[:,1] + r(z,xi)[:,0],1), \
                                 np.expand_dims(-2.*v(z,xi)[:,0] + r(z,xi)[:,1],1) ))

Psi2 = lambda z, xi: np.hstack(( np.expand_dims(r(z,xi)[:,0] + mu,1), \
                                 np.expand_dims(r(z,xi)[:,1],1) ))

Psi3 = lambda z, xi: np.hstack(( np.expand_dims(r(z,xi)[:,0] + mu - 1.,1), \
                                 np.expand_dims(r(z,xi)[:,1],1) ))


Ls = lambda z, xi: -a(z,xi) + Psi1(z,xi) \
                            - np.expand_dims((1.-mu)/R1(z,xi)**3,1) * Psi2(z,xi) \
                            - np.expand_dims(mu/R2(z,xi)**3,1)      * Psi3(z,xi)


## FORM LOSS AND JACOBIAN ***********************************************************************************
L = jit(lambda xi: np.hstack([Ls(z,xi).flatten(), Jc(z,xi)-C]))

## INITIALIZE VARIABLES *************************************************************************************
xis   = onp.zeros((H(z).shape[1],2)) # need better initialization
X     = onp.ones(1) * r_init[0]
Y     = onp.ones(1) * r_init[1]
dX    = onp.ones(1) * v_init[0]
dY    = onp.ones(1) * v_init[0]
b     = onp.ones(1) * T_init

xi = TFCDictRobust({'xis':xis,'X':X,'Y':Y,'dX':dX,'dY':dY,'b':b})


xi, iter, time = NLLS(xi,L,timer=True)

## plot: *******************************************************************************************

## compute location of L1 and L2 equilibrium points
L1 = 1. - (mu/3.)**(1./3.)
L2 = 1. + (mu/3.)**(1./3.)


p1 = MakePlot('x','y')
p1.ax[0].plot(L1,0.,'ko', markersize=2)
p1.ax[0].plot(L2,0.,'ko', markersize=2)
p1.ax[0].plot(1.-mu,0.,'ko', markersize=6)

p4.ax[0].plot(tfc['sol'][:,0,i],tfc['sol'][:,1,i],'b')

p1.ax[0].set_xlabel(r'$x$',labelpad=10)
p1.ax[0].set_ylabel(r'$y$',labelpad=10)

p1.ax[0].axis('equal')
p1.ax[0].set(ylim=(-.75, .75))
p4.ax[0].grid(True)

p4.PartScreen(4.,6.)
p4.show()