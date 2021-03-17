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
x = np.linspace(-2.,2.,N)
ind = np.argmin(np.abs(x))
H = myTfc.H

m = H(x).shape[1]

# Create the constrained expression:
K = np.array([1.,2.,3.])
g = lambda x,xi: np.dot(H(x),xi)
uslow = lambda x,n,xi: g(x,xi)+K[np.int64(n%3)]-g(np.array([0.]),xi)
u = jit(uslow)

# Run the monte carlo test
p = MakePlot(r'$x$',r'$y(x,n,g(x))$')

for k in range(nMC):
    xi = onp.random.randn(m)
    n = onp.random.rand()*10.-5.

    U = u(x,n,xi)
    val = U[ind]
    p.ax[0].plot(x,U)

    if np.round(val**3-6.*val**2+11.*val,14) != 6.:
        raise ValueError("Error on constraint is too large!")

p.ax[0].plot(np.zeros(3),K,'k',linestyle='none',markersize=10,marker='.')
p.PartScreen(8,7)
p.show()
