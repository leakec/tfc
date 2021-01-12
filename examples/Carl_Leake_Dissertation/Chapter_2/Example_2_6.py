from matplotlib import cm
from mayavi import mlab

import jax.numpy as np
from tfc.utils import egrad, MakePlot
from tfc.utils.MayaviMakePlot import MakePlot as MP

# Constants
n = 100
c1 = lambda y: y**2*np.sin(np.pi*y)

# Create grid of points
dark = np.meshgrid(np.linspace(0.,2.,n),np.linspace(0.,1.,n))
x = dark[0].flatten(); y = dark[1].flatten()

# Create the free function
g = lambda x,y: x**2*np.cos(y)+np.sin(2.*x)

# Create the constrained expression
u1 = lambda x,y: g(x,y)\
                 +(3.-2.*x)/3.*(y**2*np.sin(np.pi*y)-g(np.zeros_like(x),y))\
                 +x/3.*(np.cos(np.pi*y)-g(2.*np.ones_like(x),y)-g(np.ones_like(x),y))
du1dy = egrad(u1,1)
u = lambda x,y: u1(x,y)\
                -(y-y**2)*du1dy(x,np.zeros_like(y))\
                -y**2*(u1(x,np.ones_like(y))-u1(x,np.zeros_like(y)))

# Plot the results
U = u(x,y)

ind1 = np.where(x==0.)[0]
ind2 = np.where(y==0.)[0]
ind3 = np.where(y==1.)[0]

p = MakePlot(r'$x$',r'$y$',zlabs=r'$u(x,y,g(x,y))$')
p.ax[0].view_init(azim=-140,elev=30)
p.ax[0].plot_surface(*dark,U.reshape((n,n)),
                     cmap=cm.jet,antialiased=False,rcount=n,ccount=n)
p.ax[0].plot3D(x[ind1],y[ind1],c1(y[ind1]),'b',zorder=3,linewidth=3)
p.ax[0].plot3D(x[ind2],y[ind2],U[ind2],'m',zorder=3,linewidth=3)
p.ax[0].plot3D(x[ind3],y[ind3],U[ind2],'m',zorder=3,linewidth=3)
for k,el in enumerate(ind2):
    if not k%5:
        p.ax[0].plot3D([x[el],]*2,
                       [0.,0.05],
                       [u(x[el],0.),]*2,
                       '-k',zorder=3,linewidth=3)
p.ax[0].xaxis.labelpad = 20
p.ax[0].yaxis.labelpad = 20
p.ax[0].zaxis.labelpad = 20
p.FullScreen()
p.show()

p1 = MP()
mesh = p1.mesh(*dark,U.reshape((n,n)),colormap='jet')
mlab.axes(mesh)
p1.plot3d(x[ind1],y[ind1],c1(y[ind1]),color='b',tube_radius=0.01)
p1.plot3d(x[ind2],y[ind2],U[ind2],color='m',tube_radius=0.01)
p1.plot3d(x[ind3],y[ind3],U[ind2],color='m',tube_radius=0.01)
for k,el in enumerate(ind2):
    if not k%5:
        p1.plot3d([x[el],]*2,
                 [0.,0.1],
                 [u(x[el],0.),]*2,
                 color='k',tube_radius=0.01)
p1.show_axes = True
p1.view(azimuth=-140,elevation=30)
p1.show()

