# This is a function that solves the convection diffusion equation with TFC
# This script breaks the problem into two segments. 
# The segment lengths are chosen using nonlinear least-squares
# Hunter Johnston - Texas A&M University
# Updated: 15 Mar 2021
################################################################################
# Differential Equation
#   y'' - Pe y = 0
#
#   subject to: y(0)  = 9/10 + 1/10 exp(pi/2) (5 - 2 exp(pi/2))
#               y(pi) = exp(-pi/2)
################################################################################
from tfc import utfc
from tfc.utils import MakePlot, TFCDict, egrad, NLLS, step
import jax.numpy as np
from jax import jit, jacfwd, lax

import numpy as onp
import tqdm
from time import process_time as timer
import scipy.optimize as optim
global time, it

#: Analytical solution
soln = lambda x: (1.-np.exp(Pe*(x-1.)))/(1.-np.exp(-Pe))
dsoln = lambda x: egrad(soln,0)(x)
ddsoln = lambda x: egrad(dsoln,0)(x)

# Constants used in the differential equation:
Pe = 10**6
tol = 1e-13

xI = 0.
xf = 1.
yi = 1.
yf = 0.
xpBound = 1.-1.*10**-6

# Create the ToC Class:
N = 200
c = 1.
m = 190
nC = 3
tfc = utfc(N,nC,m,basis='CP',x0=-1,xf=1.)

# Get the Chebyshev polynomials
H = tfc.H
dH = tfc.dH
H0 = H(tfc.z[0])
Hf = H(tfc.z[-1])
Hd0 = dH(tfc.z[0])
Hdf = dH(tfc.z[-1])

# Create the constraint expression and its derivatives
z = tfc.z
c1 = lambda xp: 2./(xp-xI)
c2 = lambda xp: 2./(xf-xp)

x1 = lambda z,xp: (z+1.)/c1(xp)+xI
x2 = lambda z,xp: (z+1.)/c2(xp)+xp

phi1_s1 = lambda a: (1.-2.*a+a**2)/4.
phi2_s1 = lambda a: (3.+2.*a-a**2)/4.
phi3_s1 = lambda a: (-1.+a**2)/2.

phi1_s2 = lambda a: (3.-2.*a-a**2)/4.
phi2_s2 = lambda a: (1.-a**2)/2.
phi3_s2 = lambda a: (1.+2.*a+a**2)/4.

y1 = lambda z,xi,xp: \
np.dot(H(z),xi['xi1'])+phi1_s1(z)*(yi               -np.dot(H0, xi['xi1']))\
                      +phi2_s1(z)*(xi['y']          -np.dot(Hf, xi['xi1']))\
                      +phi3_s1(z)*(xi['yd']/c1(xp)  -np.dot(Hdf,xi['xi1']))
ydz1 = egrad(y1,0)
yddz1 = egrad(ydz1,0)


yd1 = lambda z,xi,xp: ydz1(z,xi,xp)*c1(xp)
ydd1 = lambda z,xi,xp: yddz1(z,xi,xp)*c1(xp)**2

y2 = lambda z,xi,xp: \
np.dot(H(z),xi['xi2'])+phi1_s2(z)*(xi['y']         -np.dot(H0, xi['xi2']))\
                      +phi2_s2(z)*(xi['yd']/c2(xp) -np.dot(Hd0,xi['xi2']))\
                      +phi3_s2(z)*(yf              -np.dot(Hf, xi['xi2']))

ydz2 = egrad(y2,0)
yddz2 = egrad(ydz2,0)
yd2 = lambda z,xi,xp: ydz2(z,xi,xp)*c2(xp)
ydd2 = lambda z,xi,xp: yddz2(z,xi,xp)*c2(xp)**2

L1 = lambda z,xi,xp: ydd1(z,xi,xp)-Pe*yd1(z,xi,xp)
L2 = lambda z,xi,xp: ydd2(z,xi,xp)-Pe*yd2(z,xi,xp)

L = jit(lambda z,xi,xp: np.hstack(( L1(z,xi,xp), L2(z,xi,xp) )), static_argnums=[0,])

def Jdark(x,xi,xp):
    jacob = jacfwd(L,1)(z,xi,xp)
    return np.hstack((jacob[k] for k in xi.keys()))
J = jit(lambda z,xi,xp: Jdark(z,xi,xp),static_argnums=[0,])


# Create the residual and jacobians
xi1  = onp.zeros(H(z).shape[1])
xi2  = onp.zeros(H(z).shape[1])
y    = onp.zeros(1)
yd   = onp.zeros(1)

xi = TFCDict({'xi1':xi1,'xi2':xi2,'y':y,'yd':yd})

# Create the NLLS
def cond(val):
    return np.all(np.array([
                np.max(np.abs(L(z,val['xi'],val['xp']))) > tol,
                val['it'] < 30,
                np.max(np.abs(val['dxi'])) > tol]))
def body(val):
    val['dxi'] = -np.dot(np.linalg.pinv(J(z,val['xi'],val['xp'])),L(z,val['xi'],val['xp']))
    val['xi'] += val['dxi']
    val['it'] += 1
    return val

nlls = jit(lambda val: lax.while_loop(cond,body,val))

xp = 0.75
val = {'xi':xi,'dxi':np.ones_like(xi.toArray()),'it':0,'xp':xp}
nlls(val)

time = 0.
it = 0.
# Find the value of xp using a genetic algorithm
def fMin(xp, val=val, nlls=nlls):
    global time, it

    # Run the least-squares
    val['xp'] = xp

    if it == 0:
        val = nlls(val)
    elif it == 1:
        tic = timer()
        val = nlls(val)
        val['dxi'].block_until_ready()
        toc = timer()
        time += 2.*(toc-tic)
    else:
        tic = timer()
        val = nlls(val)
        val['dxi'].block_until_ready()
        toc = timer()
        time += (toc-tic)

    it += 1

    return np.max(np.abs(L(z,val['xi'],val['xp'])))

Soln = optim.differential_evolution(fMin,[(0.75,xpBound)],maxiter=4)
xp = Soln['x']
print(Soln.message)

val['xp'] = xp
val = nlls(val)
xi = val['xi']


X = np.hstack((x1(z,xp), x2(z,xp)))
Y = np.hstack((y1(z,xi,xp), y2(z,xi,xp) ))


# p1 = MakePlot(onp.array([['x (m)']]),onp.array([['y (m)']]))
# p1.ax[0].plot(X,Y,label='TFC Solution')
# p1.ax[0].plot(X,soln(X),label='Analytical Solution')
# p1.ax[0].legend()
# p1.show()

print('{:.2e} & {:.2e} & {:.5f} & {:.2f}'.format(np.max(np.abs(Y - soln(X))), np.max(np.abs(L(z,xi,xp))), xp[0], time ))
