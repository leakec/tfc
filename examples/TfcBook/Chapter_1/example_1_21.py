import numpy as onp
import jax.numpy as np
from jax import jit

from tfc import utfc
from tfc.utils import MakePlot, step


# Constants:
N = 101
m = 8
nC = -1

nMC = 100

# Create the TFC class:
myTfc = utfc(N,nC,m,x0=-2.,xf=2.,basis='LeP')
x = np.linspace(0.,2.,N)
ind = np.argmin(np.abs(x-1.))
H = myTfc.H

m = H(x).shape[1]

# Create the constrained expression:
g = lambda x,xi: np.dot(H(x),xi)
uslow = lambda x,n,xi: g(x,xi)+np.int64(n)*np.pi-g(np.array([1.]),xi)
u = jit(uslow)

# Run the monte carlo test
p = MakePlot(r'$x$',r'$y(x,n,g(x))$')

for k in range(nMC):
    xi = onp.random.randn(m)
    n = onp.random.rand()*10.-5.

    U = u(x,n,xi)
    val = U[ind]
    p.ax[0].plot(x,U)

p.ax[0].plot(np.ones(9),np.linspace(-4.,4.,9)*np.pi,'k',linestyle='none',markersize=10,marker='.')
p.ax[0].set_ylim([-15,15])
p.ax[0].set_xlim([0.,2.])
p.ax[0].grid(True)
p.PartScreen(8,7)
p.show()
