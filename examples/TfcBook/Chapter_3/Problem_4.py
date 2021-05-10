import jax.numpy as np
from tfc import utfc
from tfc.utils import egrad, MakePlot, NLLS

# Constants:
n = 100
nC = 2
m = 60
th0 = 0. 
thf = 1.3
r0 = 0.
rf = 5.

# Create TFC class:
myTfc = utfc(n,nC,m,x0=th0,xf=thf)
th = myTfc.x
H = myTfc.H

# Create constrained expression:
g = lambda th,xi: np.dot(H(th),xi)
r = lambda th,xi: g(th,xi)+\
                  (th-thf)/(th0-thf)*(r0-g(th0*np.ones_like(th),xi))+\
                  (th-th0)/(thf-th0)*(rf-g(thf*np.ones_like(th),xi))

# Create loss function:
dr = egrad(r)
d2r = egrad(dr)
L = lambda xi: -r(th,xi)**2*(dr(th,xi)*np.tan(th)+2.*d2r(th,xi))+\
               -np.tan(th)*dr(th,xi)**3+3.*r(th,xi)*dr(th,xi)**2+r(th,xi)**3

# Solve the problem:
xi = np.zeros(H(th).shape[1])
xi,_,time = NLLS(xi,L,timer=True)

# Print out statistics:
print("Solution time: {0} seconds".format(time))

# Plot the solution and residual
p = MakePlot([r"$y$"],[r"$x$"])
p.ax[0].plot(r(th,xi)*np.sin(th),r(th,xi)*np.cos(th),"k")
p.ax[0].axis("equal")
p.ax[0].grid(True)
p.ax[0].invert_yaxis()
p.PartScreen(8,7)
p.show()

p2 = MakePlot([r"$\theta$"],[r"$L$"])
p2.ax[0].plot(th,np.abs(L(xi)),"k",linestyle="None",marker=".",markersize=10)
p2.ax[0].set_yscale("log")
p2.PartScreen(8,7)
p2.show()

