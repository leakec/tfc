import numpy as np
from matplotlib import cm
from tfc.utils import MakePlot

# Constants 
n = 100

# Create the domain
dark = np.meshgrid(np.linspace(-2.,2.,n),np.linspace(-2.,2.,n))
x = dark[0].flatten(); y = dark[1].flatten()

# Free functions
gu = lambda x,y: x*y+np.sin(x)+y**2
gv = lambda x,y: x**2*y*np.cos(y)*np.exp(x)

intgu = lambda x,y: y**2/2.+np.sin(1.)*y+y**3/3.
int1 = intgu(2.,2.)-intgu(2.,-1.)

# Constrainted expressions
u1 = lambda x,y: gu(x,y)+(1.-x)*(np.cos(np.pi*y)-gu(np.zeros_like(x),y))+x/3.*(np.exp(1.)-int1)
u = lambda x,y: u1(x,y)+(1.-2.*y)/2.*(u1(x,2.*np.ones_like(y))-u1(x,np.ones_like(y))-2.)

v1 = lambda x,y: gv(x,y)+5.-np.cos(np.pi*y)-gv(np.zeros_like(x),y)
v = lambda x,y: v1(x,y)+5.-v1(x,np.zeros_like(y))-u(x,np.zeros_like(y))

# Plot results
U = u(x,y).reshape((n,n))
V = v(x,y).reshape((n,n))
p = [MakePlot(r"$x$",r"$y$",zlabs=r"$u(x,y,g^u(x,y))$"),MakePlot(r"$x$",r"$y$",zlabs=r"$v(x,y,g^v(x,y),g^u(x,y))$")]
p[0].ax[0].plot_surface(*dark,U,cmap=cm.gist_rainbow,antialiased=False,rcount=n,ccount=n)
p[0].ax[0].plot(np.zeros_like(x),y,np.cos(np.pi*y),'k',linewidth=4.,zorder=3)
p[1].ax[0].plot_surface(*dark,V,cmap=cm.gist_rainbow,antialiased=False,rcount=n,ccount=n)
p[1].ax[0].plot(np.zeros_like(x),y,5.-np.cos(np.pi*y),'k',linewidth=4.,zorder=3)
for k in range(2):
    p[k].ax[0].xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    p[k].ax[0].yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    p[k].ax[0].zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    p[k].ax[0].xaxis.labelpad = 20
    p[k].ax[0].yaxis.labelpad = 20
    p[k].ax[0].zaxis.labelpad = 15
    p[k].ax[0].view_init(30.,-135.)
    p[k].PartScreen(8,7)
    p[k].show()
