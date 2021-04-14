# This script solves Problem #7 of Chapter 3 in the TFC book
####################################################################################################
# Differential Equations
#   ẍ =  2ẏ + x - (x + μ)*(1 - μ)/(R₁)³ - (x + μ - 1)*μ/(R₂)³
#   ÿ = -2ẋ + y -  y     *(1 - μ)/(R₁)³ -  y         *μ/(R₂)³   
    #   C = x² + y² + 2(1 - μ )/R₁ + 2μ/R₂ + μ(1 - μ) - (ẋ + ẏ)

#   subject to: x(0) = x(T),    ẋ(0) = ẋ(T)
#               y(0) = y(T),    ẏ(0) = ẏ(T)
#
#   Note: The scaling dr/dt = dr/dz * dz/dζ * dζ/dt = b²/R₂ dr/dz
#       [0, T]   →   [0, ζf]   →   [z0 , zf]
#       problem      scaled       basis [-1, +1]
####################################################################################################
from tfc import utfc
from tfc.utils import TFCDictRobust, egrad, NLLS, MakePlot
import jax.numpy as np
import numpy as onp
####################################################################################################

## constants: **************************************************************************************
mu = 0.01215    # gravitation parameter of the Earth-Moon system
C  = 3.05       # specified jacobi constant level

## initial guess values: ***************************************************************************
r_init = [0.79, 0.]  
v_init = [0.,   0.42] 
T_init = 21.4
# !!! These initial guesses comes from the Richardson's third-order analytical method for Halo-type
#     periodic orbits. See: http://articles.adsabs.harvard.edu/full/1980CeMec..22..241R

## user defined parameters: ************************************************************************
N       = 140   # number of discretization points
m       = 130   # number of basis function terms
basis   = 'CP'  # basis function type
# !!! Here we can see that I large number of basis terms is used. This is needed for this specific
#     problem to obtain the desirable accuracy

## construct univariate tfc class: *****************************************************************
tfc  = utfc(N, 4, m, basis=basis, x0=-1., xf=1.)
# !!! Note that here I didn't explicitly define nC = 4, but I had inluded the number of constraints
#     in my call to the utfc class

H = tfc.H
dH = tfc.dH

H0 = H(tfc.z[0])
Hf = H(tfc.z[-1])

Hp0 = dH(tfc.z[0])
Hpf = dH(tfc.z[-1])

## defined the constrained expressions: ************************************************************
z = tfc.z
z0 = z[0]
zf = z[-1]

## defined some useful terms
R0 = lambda xi: np.array([xi['X'],xi['Y']]).flatten()
D0 = lambda xi: np.sqrt( (R0(xi)[0]+mu-1.)**2 + R0(xi)[1]**2 )
V0 = lambda xi: np.array([xi['dX'],xi['dY']]).flatten()/xi['b']**2/(1./D0(xi))
# !!! The terms R0, D0, and V0 are useful in the solution of the unknowns arising from the relative
#     constraints. For example, R0 contains the value of the initial position [x₀, y₀]

# switching functions built from sⱼ(x) = [1, x, x² , x³] and sⱼ(y) = [1, y, y² , y³] 
phi1 = lambda a:\
    np.expand_dims(1./(zf-z0)**3 * (-zf**2*(3.*z0-zf) + 6.*z0*zf*a - 3.*(z0+zf)*a**2 + 2.*a**3),1)
phi2 = lambda a:\
    np.expand_dims(1./(zf-z0)**3 * (-z0**2*(z0-3.*zf) - 6.*z0*zf*a + 3.*(z0+zf)*a**2 - 2.*a**3),1)
phi3 = lambda a:\
    np.expand_dims(1./(zf-z0)**2 * (-z0*zf**2 + zf*(2.*z0+zf)*a - (z0 + 2.*zf)*a**2 + a**3),1)
phi4 = lambda a:\
    np.expand_dims(1./(zf-z0)**2 * (-z0**2*zf + z0*(z0+2.*zf)*a - (2.*z0 + zf)*a**2 + a**3),1)

# form constrained expressions
r = lambda z, xi: np.dot(H(z),xi['xis']) + phi1(z)*(R0(xi) - np.dot(H0, xi['xis'])) \
                                         + phi2(z)*(R0(xi) - np.dot(Hf, xi['xis'])) \
                                         + phi3(z)*(V0(xi) - np.dot(Hp0,xi['xis'])) \
                                         + phi4(z)*(V0(xi) - np.dot(Hpf,xi['xis']))

# terms defining distance from third-body to the primary (R1) and the secondary (R2)
R1 = lambda z, xi: np.sqrt( (r(z,xi)[:,0]+mu   )**2 + r(z,xi)[:,1]**2 ) 
R2 = lambda z, xi: np.sqrt( (r(z,xi)[:,0]+mu-1.)**2 + r(z,xi)[:,1]**2 )

# scaling term dr/dt = dr/dz * dz/dζ * dζ/dt = b²/R₂ dr/dz
d2 = lambda z,xi: 1./R2(z,xi)
# !!! where t is the problem domain, ζ is the scaled domain, and z is the basis domain

# derivatives of the constrained expressions
v = lambda z,xi: np.expand_dims(d2(z,xi),1) * xi['b']**2 * egrad(r)(z,xi)
a = lambda z,xi: np.expand_dims(d2(z,xi),1) * xi['b']**2 * egrad(v)(z,xi)


## form the loss vector: ***************************************************************************

# jacobi constant equation
Jc = lambda Z, xi: (r(z,xi)[:,0]**2 + r(z,xi)[:,1]**2) \
                               + 2.*(1.-mu)/R1(z,xi) \
                               + 2.*mu/R2(z,xi) \
                               + (1.-mu)*mu \
                               - (v(z,xi)[:,0]**2 + v(z,xi)[:,1]**2)

# terms of the differential equation
Psi1 = lambda z, xi: np.hstack(( np.expand_dims( 2.*v(z,xi)[:,1] + r(z,xi)[:,0],1), \
                                 np.expand_dims(-2.*v(z,xi)[:,0] + r(z,xi)[:,1],1) ))

Psi2 = lambda z, xi: np.hstack(( np.expand_dims(r(z,xi)[:,0] + mu,1), \
                                 np.expand_dims(r(z,xi)[:,1],1) ))

Psi3 = lambda z, xi: np.hstack(( np.expand_dims(r(z,xi)[:,0] + mu - 1.,1), \
                                 np.expand_dims(r(z,xi)[:,1],1) ))

# loss vector of the differential equations
Ls = lambda z, xi: -a(z,xi) + Psi1(z,xi) \
                            - np.expand_dims((1.-mu)/R1(z,xi)**3,1) * Psi2(z,xi) \
                            - np.expand_dims(mu/R2(z,xi)**3,1)      * Psi3(z,xi)

# loss function
L = lambda xi: np.hstack([Ls(z,xi).flatten(), Jc(z,xi)-C])
# !!! Notice the Ls(z,xi).flatten(), this is needed because Ls is a [N,2] vector and the loss vector
#     must be [dim,]. Also, the second term is the algebraic equation enforcing the specific Jacobi
#     constant level (C = 3.05)


## initialize variables: ***************************************************************************
xis   = onp.zeros((H(z).shape[1],2))

# specify first two terms of the Chebyshev expansion, i.e. the 4th- and 5th-order terms
xis[0:2,0] = [-3.e-03,   9.e-12]
xis[0:2,1] = [ 2.e-12,   1.e-02]
# !!! This problem is more difficult and therefore, a better initial guess then all zeros is needed
#     when using Chebyshev polynomials

# initialization of other unknown variables based on lines 29-31
X     = onp.ones(1) * r_init[0]
Y     = onp.ones(1) * r_init[1]
dX    = onp.ones(1) * v_init[0]
dY    = onp.ones(1) * v_init[1]
b     = onp.ones(1) * np.sqrt(2./T_init)

# create a TFC dictionary with the unknowns
xi = TFCDictRobust({'xis':xis,'X':X,'Y':Y,'dX':dX,'dY':dY,'b':b})

## solving the system of equations: ****************************************************************
xi, iter, time = NLLS(xi,L,timer=True)

## plotting: ***************************************************************************************

# compute location of L1 and L2 equilibrium points
L1 = 1. - (mu/3.)**(1./3.)
L2 = 1. + (mu/3.)**(1./3.)

p1 = MakePlot('x','y')
p1.ax[0].plot(L1,0.,'ko', markersize=2)
p1.ax[0].plot(L2,0.,'ko', markersize=2)
p1.ax[0].plot(1.-mu,0.,'ko', markersize=6)

p1.ax[0].plot(r(z,xi)[:,0], r(z,xi)[:,1])

p1.ax[0].set_xlabel(r'$x$',labelpad=10)
p1.ax[0].set_ylabel(r'$y$',labelpad=10)

p1.ax[0].axis('equal')
p1.ax[0].set(ylim=(0.8, 1.2))
p1.ax[0].set(ylim=(-.3, .3))
p1.ax[0].grid(True)

p1.PartScreen(7.,6.)
p1.show()