from tfc import utfc
from tfc.utils import TFCDict, egrad, NLLS, MakePlot

import numpy as onp
import jax.numpy as np
from jax import vmap, jacfwd, jit, lax

import scipy.optimize as optim


import tqdm
from time import process_time as timer

from scipy.integrate import simpson

## TEST PARAMETERS: ***************************************************
tol = np.finfo(float).eps
iterMax = 50

## CONSTANTS: *********************************************************
x0 = 0.
xf = 1.

alfa = 1.
beta = 1.

# Number of points to use
N = 35

# Number of basis functions to use
m = 30

# Number of constraints
nCx = 2
nCu = 0


## GET CHEBYSHEV VALUES: *********************************************
xtfc = utfc(N,nCx,m,basis='CP', x0 = -1, xf = 1.)
utfc = utfc(N,nCu,m,basis='CP', x0 = -1, xf = 1.)

Hx  = xtfc.H
Hx0 = Hx(xtfc.z[0:1])
Hxf = Hx(xtfc.z[-1:])

Hu  = utfc.H

## DEFINE THE ASSUMED SOLUTION: *************************************
z = xtfc.z
z0 = z[0]
zf = z[-1]

phi1 = lambda a: (zf - a) / (zf - z0)
phi2 = lambda a: (a - z0) / (zf - z0)

x  = lambda z, xi: np.dot(Hx(z),xi) + phi1(z)*(x0 - np.dot(Hx0,xi)) + phi2(z)*(xf - np.dot(Hxf,xi))
xp  = egrad(x,0)

u  = lambda z,xi: np.dot(Hu(z),xi)
up  = egrad(u,0)


## LOSS FUNCTIONS AND JACOB *****************************************
Lx = lambda z, xi: -xi['b']**2 * xp(z,xi['xi_x']) - alfa*x(z,xi['xi_x']) - beta*u(z,xi['xi_u'])
Lu = lambda z, xi: -xi['b']**2 * up(z,xi['xi_u']) - beta*x(z,xi['xi_x']) + alfa*u(z,xi['xi_u'])
H = lambda z,xi: 0.5*x(z,xi['xi_x'])**2 - 0.5*u(z,xi['xi_u'])**2 - alfa/beta * x(z,xi['xi_x'])*u(z,xi['xi_u'])
Lf = lambda z, xi: H(z,xi)[-1]

L = lambda xi: np.hstack(( Lx(z,xi), Lu(z,xi), H(z,xi) ))


xi_x = onp.zeros(Hx(z).shape[1])
xi_u = onp.zeros(Hu(z).shape[1])
b    = onp.ones(1) * np.sqrt(2.)

xi = TFCDict({'xi_x':xi_x,'xi_u':xi_u,'b':b})

## SOLVE THE SYSTEM *************************************************
xi,it,time = NLLS(xi,L,timer=True,tol=tol,maxIter=iterMax)

t = (z-z[0])/xi['b']**2

X = x(z,xi['xi_x'])
U = u(z,xi['xi_u'])

Ham = onp.zeros(len(t))
int = onp.zeros(len(t))
for i in range(0,len(t)):
    int[i] = 0.5 * (X[i]**2 + U[i]**2)
    Ham[i] = int[i] + -U[i]/beta * (alfa*X[i] + beta*U[i])

cost = simpson(int,x=t)

tf = 2./xi['b']**2

print('{:.2e} & {:.2e} & {:.8f} & {:.5f} & {:d} & {:.2f}'.format(np.max(np.abs(L(xi))), np.max(np.abs(H(z,xi))), cost, tf.tolist()[0], it, time*1000 ))

# Plots
MS = 12

p1 = MakePlot(onp.array([['t','t']]),onp.array([[r'$x(t)$',r'$y(t)$']]))
p1.fig.subplots_adjust(wspace=0.25, hspace=0.25)
p1.ax[0].plot(t,x(z,xi['xi_x']),label='x(t)', linewidth = 2)
p1.ax[1].plot(t,u(z,xi['xi_u']),label='y(t)', linewidth = 2)
p1.ax[0].grid(True)
p1.ax[1].grid(True)
p1.FullScreen()
p1.show()
# p1.save('figures/unknownTimeStates')

p2 = MakePlot('t',r'$|Loss|$')
p2.ax[0].plot(t,onp.abs(Lx(z,xi)),'r*', markersize = MS, label='|$L_x(t)$|')
p2.ax[0].plot(t,onp.abs(Lu(z,xi)),'kx', markersize = MS, label='|$L_u(t)$|')
p2.ax[0].plot(t,onp.abs(H(z,xi)), 'b+', markersize = MS, label='|$H(t)$|')
p2.ax[0].set_yscale('log')
p2.ax[0].legend()
p2.ax[0].grid(True)
p2.PartScreen(7.,6.)
p2.show()
# p2.save('figures/unknownTimeLoss')
