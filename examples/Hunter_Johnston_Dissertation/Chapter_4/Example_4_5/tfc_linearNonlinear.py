# This is a function that solves the linear-nonlinear differential
# equation sequence with TFC
##################################################################
# Differential Equation
#   y'' + yy'^a = exp(π/2) - exp(π/2 - x)
#
#   subject to: y(0)  = 9/10 + 1/10 exp(π/2) [5 - 2 exp(π/2)]
#               y(π) = exp(-π/2)
###################################################################
from tfc import utfc
from tfc.utils import MakePlot, TFCDict, egrad, NLLS
import jax.numpy as np
from jax import jit

import numpy as onp
import tqdm
##################################################################
N = 100
m = 20
basis = 'CP'
tol = 1e-16
iterMax = 50

## Boundaries: ***************************************************
x0 = 0.
x1 = np.pi/2.
xf = np.pi

## Initial Conditions: *******************************************************
y0  = 9./10. + 1./10. * np.exp(np.pi/2.) * (5. - 2. * np.exp(np.pi/2.))
yf  = np.exp(-np.pi/2.)


nC  = 3 # number of constraints

## Compute true solution
def ytrue(a):
    val = onp.zeros_like(a)
    for i in range(0,len(a)):
        if a[i] <= np.pi/2.:
            val[i] = - 1./5. * np.exp(np.pi - 2.*a[i]) \
                     + 1./2. * np.exp(np.pi/2. - a[i]) \
                     + (9.*np.cos(a[i]) + 7.*np.sin(a[i])) / 10.
        else:
            val[i] = np.exp(np.pi/2. - a[i])
    return val


## GET CHEBYSHEV VALUES: *********************************************

# First segment
tfc1 = utfc(N,nC,m,basis = basis, x0 = x0, xf = x1)
xs1 = tfc1.x

Hs1 = tfc1.H
dHs1 = tfc1.dH

H0s1 = Hs1(tfc1.x[0])
Hfs1 = Hs1(tfc1.x[-1])

Hfps1 = dHs1(tfc1.x[-1])

# Second segment
tfc2 = utfc(N,nC,m,basis = basis, x0 = x1, xf = xf)
xs2 = tfc2.x

Hs2 = tfc2.H
dHs2 = tfc2.dH

H0s2 = Hs2(tfc2.x[0])
Hfs2 = Hs2(tfc2.x[-1])

H0ps2 = dHs2(tfc2.x[0])


## DEFINE THE ASSUMED SOLUTION: *************************************

# First segment
phi11 = lambda a: 1./(x1 - x0)**2 * (x1**2 - 2.*x1*a + a**2)
phi12 = lambda a: 1./(x1 - x0)**2 * (x0*(x0-2.*x1) \
                                        + 2.*x1*a - a**2)
phi13 = lambda a: 1./(x1 - x0)    * (x0*x1 - (x0+x1)*a + a**2)

ys1 = lambda x, xi: np.dot(Hs1(x),xi['xi1']) \
        + phi11(x)*(y0 - np.dot(H0s1, xi['xi1'])) \
        + phi12(x)*(xi['y1']  - np.dot(Hfs1, xi['xi1'])) \
        + phi13(x)*(xi['y1d'] - np.dot(Hfps1,xi['xi1']))
yps1  = egrad(ys1,0)
ypps1 = egrad(yps1,0)


# Second segment
phi21 = lambda a: 1./(xf - x1)**2 * (xf*(xf-2.*x1) \
                                    + 2.*x1*a - a**2)
phi22 = lambda a: 1./(xf - x1)    * (-xf*x1 + (xf + x1)*a - a**2)
phi23 = lambda a: 1./(xf - x1)**2 * (x1**2 - 2.*x1*a + a**2)

ys2 = lambda x, xi: np.dot(Hs2(x),xi['xi2']) \
        + phi21(x)*(xi['y1']  - np.dot(H0s2,xi['xi2'])) \
        + phi22(x)*(xi['y1d'] - np.dot(H0ps2,xi['xi2'])) \
        + phi23(x)*(yf        - np.dot(Hfs2,xi['xi2']))
yps2  = egrad(ys2,0)
ypps2 = egrad(yps2,0)

## DEFINE LOSS AND JACOB ********************************************
f = lambda x: -np.exp(np.pi - 2.*x) + np.exp(np.pi/2. - x)


L1 = lambda xi: ypps1(xs1, xi) + ys1(xs1, xi) - f(xs1)
L2 = lambda xi: ypps2(xs2, xi) + ys2(xs2, xi)*yps2(xs2, xi) \
                               - f(xs2)

L = jit( lambda xi: np.hstack(( L1(xi), L2(xi))) )



## SOLVE THE SYSTEM *************************************************
xi1 =  onp.zeros(Hs1(xs1).shape[1])
xi2 =  onp.zeros(Hs2(xs2).shape[1])

m = (yf-y0)/(xf-x0)
y1  =  onp.ones(1) * (m*xs1[-1] + y0)
y1d =  onp.ones(1) * m

xi0 = TFCDict({'xi1':xi1,'xi2':xi2,'y1':y1,'y1d':y1d})
xi = TFCDict({'xi1':xi1,'xi2':xi2,'y1':y1,'y1d':y1d})

xi,it,time = NLLS(xi,L,timer=True)


## COMPUTE ERROR AND RESIDUAL ***************************************
x = np.hstack((xs1,xs2))

yinit = np.hstack(( ys1(xs1,xi0), ys2(xs2,xi0) ))
y = np.hstack(( ys1(xs1,xi), ys2(xs2,xi) ))
yp = np.hstack(( yps1(xs1,xi), yps2(xs2,xi) ))


err = onp.abs(y - ytrue(x))
res = onp.abs(L(xi))

print()
print('Max Error: '             + str(np.max(err)))
print('Max Loss: '              + str(np.max(res)))
print('Computation time [ms]: ' + str(time*1000))
print()

## Plots
#Plot 1
p1 = MakePlot([[r'$x$',r'$x$',r'$x$']],[[r'$y(x)$',r'$y_x(x)$',r'$y_{xx}(x)$']])
p1.ax[0].plot(x,y,'k',label=r'$y(x)$', linewidth = 2)

p1.ax[1].plot(x,yp,'k',label=r'$y_x(x)$', linewidth = 2)

p1.ax[2].plot(xs1,ypps1(xs1,xi), 'k', label=r'$y_{xx}(x)$', linewidth = 2)
p1.ax[2].plot(xs2,ypps2(xs2,xi), 'k', linewidth = 2)

p1.ax[0].grid('True')
p1.ax[1].grid('True')
p1.ax[2].grid('True')

p1.fig.subplots_adjust(wspace=0.75)

p1.PartScreen(10.,6.)
p1.show()
# p1.save('linearNonlinear_function')

#Plot 2
p2 = MakePlot(r'$x$',r'$|\mathbb{L}(\mathbf{\xi})|$')
p2.ax[0].plot(x,err,'k*',label=r'$y(x)$', linewidth = 2)

p2.ax[0].grid('True')
p2.ax[0].set_yscale('log')


p2.PartScreen(7.,6.)
p2.show()
# p2.save('linearNonlinear_error')

#Plot 3
p3 = MakePlot(r'$x$',r'$y(x)$')
p3.ax[0].plot(x,y,'k',label=r'True Solution', linewidth = 2)
p3.ax[0].plot(x,yinit,'r--',label=r'Initialization', linewidth = 2)
p3.ax[0].grid('True')
p3.ax[0].legend(framealpha=0.5)

p3.PartScreen(7.,6.)
p3.show()
# p3.save('linearNonlinear_init')
