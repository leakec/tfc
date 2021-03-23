# hunter comment to see if symlink works
import jax.numpy as np
from jax import jit
import numpy as onp

from tfc import utfc
from tfc.utils import MakePlot, step

onp.random.seed(0)

# Constants
numFuncs = 20
m = 5
n = 250

# Boundaries
ub = lambda x: x/30.*np.sin(2.*x) + 0.2
lb = lambda x: x**2/np.exp(x)*np.cos(x) - 0.3

# Get CP
myTfc = utfc(n,-1,m,basis='CP',x0=0.,xf=10.)
H = myTfc.H
x = np.linspace(0.,8.,n)
m = H(x).shape[1]

# Create inequality only plot
g = lambda x,xi: np.dot(H(x),xi)
u = jit(lambda x,xi: g(x,xi)+(ub(x)-g(x,xi))*step(g(x,xi)-ub(x))+(lb(x)-g(x,xi))*step(lb(x)-g(x,xi)))
def antiu(x,xi):
    dark = onp.array(g(x,xi))
    ind = onp.where(np.logical_and(dark>lb(x),dark<ub(x)))[0]
    dark[ind] = np.nan
    return dark

p = MakePlot(r'$x$',r'$y(x,g(x))$')
for k in range(numFuncs):
    xi = onp.random.rand(m)
    pl = p.ax[0].plot(x,u(x,xi))[0]
    color = pl.get_color()
    #p.ax[0].plot(x,antiu(x,xi),'--',color=color) # Uncomment to see original free function as a dashed line
p.ax[0].plot(x,ub(x),'k',linewidth=3,linestyle='--')
p.ax[0].plot(x,lb(x),'k',linewidth=3,linestyle='--')
p.ax[0].set_ylim([-1.,0.5])
p.PartScreen(8,7)
p.show()

# Inequality with linear constraints
onp.random.seed(0)

# Constants
numFuncs = 20
m = 10

# Get CP
myTfc = utfc(n,0,m,basis='LeP',x0=0.,xf=10.)
H = myTfc.H
m = H(x).shape[1]

H0 = H(np.array([0.]))
H4 = H(np.array([4.]))
H8 = H(np.array([8.]))

# Create the constrained expression
g = lambda x,xi: np.dot(H(x),xi)+\
                 (x**2-12.*x+32.)/32.*(0.-np.dot(H0,xi))+\
                 (-x**2+8.*x)/16.*(-0.2-np.dot(H4,xi))+\
                 (x**2-4.*x)/32.*(-0.1-np.dot(H8,xi))

u = jit(lambda x,xi: g(x,xi)+(ub(x)-g(x,xi))*step(g(x,xi)-ub(x))+(lb(x)-g(x,xi))*step(lb(x)-g(x,xi)))

def antiu(x,xi):
    dark = onp.array(g(x,xi))
    ind = onp.where(np.logical_and(dark>lb(x),dark<ub(x)))[0]
    dark[ind] = np.nan
    return dark

# Generate the plot
p1 = MakePlot(r'$x$',r'$y(x,g(x))$')
for k in range(numFuncs):
    xi = onp.random.rand(m)
    pl = p1.ax[0].plot(x,u(x,xi))[0]
    color = pl.get_color()
    #p1.ax[0].plot(x,antiu(x,xi),'--',color=color) # Uncomment to see original free function as a dashed line
p1.ax[0].plot(x,ub(x),'k',linewidth=3,linestyle='--')
p1.ax[0].plot(x,lb(x),'k',linewidth=3,linestyle='--')
p1.ax[0].plot(0.,0.,'k',linestyle=None,markersize=14,marker='.')
p1.ax[0].plot(4.,-0.2,'k',linestyle=None,markersize=14,marker='.')
p1.ax[0].plot(8.,-0.1,'k',linestyle=None,markersize=14,marker='.')
p1.ax[0].set_ylim([-1.,0.5])
p1.PartScreen(8,7)
p1.show()

