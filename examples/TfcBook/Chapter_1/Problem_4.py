import jax.numpy as np
from tfc.utils import MakePlot

# Constants:
n = 300

# Domain:
t = np.linspace(0.,2.,n)

# Constrained expressions:
u = lambda t,g: g(t)+\
                (t-1.)*(t-2.)/2.*(0.-g(0.))+\
                -t*(t-2.)*(np.pi-g(1.))+\
                t*(t-1.)/2.*(np.exp(1.)-g(2.))

v = lambda t,g: g(t)+\
                (t-1.)*(t-2.)/2.*(0.-g(0.))+\
                -t*(t-2.)*(2.-g(1.))+\
                t*(t-1.)/2.*(-3.-g(2.))

# Create free functions:
gu1 = lambda t: np.sin(10.*t)
gv1 = lambda t: np.cos(7.*t)
gu2 = lambda t: t**2+t+5.
gv2 = lambda t: np.exp(t)/(1.+t)
gu3 = lambda t: t%1
gv3 = lambda t: np.cos(3.*np.sqrt(t))*t

# Create the plot:
p = MakePlot(r"u(t)",r"v(t)")
p.ax[0].plot(u(t,gu1),v(t,gv1),"r")
p.ax[0].plot(u(t,gu2),v(t,gv2),"g")
p.ax[0].plot(u(t,gu3),v(t,gv3),"b")
p.ax[0].plot([0.,np.pi,np.exp(1.)],
             [0.,2.,-3.],
             "k",linestyle="None",
             marker=".",markersize=10)
p.FullScreen()
p.show()

