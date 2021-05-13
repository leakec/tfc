from matplotlib import cm
from mayavi import mlab

import numpy as onp
import jax.numpy as np
from jax import jit
from tfc.utils import egrad, MakePlot
from tfc.utils.MayaviMakePlot import MakePlot as MP

# Constants
n = 100

# Create grid of points
dark = np.meshgrid(np.linspace(0.,1.,n),np.linspace(0.,1.,n))
darkV = [dark[0]+2.,dark[1]]
x = dark[0].flatten(); y = dark[1].flatten()

# Create the free functions
gu = lambda x,y,c: 3.*x/10.*np.cos(9.*y)*np.cos(c['a']*x)+x**2/2.
gv = lambda x,y,c: y**3+c['b']*np.cos(4.*x**2)
intgv = lambda x,y,c: 1./4.+c['b']*np.cos(4.)

# Create the constrained expression
v1 = lambda x,y,c: gv(x,y,c)\
                   + 2.-intgv(x,y,c)
vslow = lambda x,y,c: v1(x,y,c)\
                      +(y-0.5)*(v1(x,np.zeros_like(y),c)-v1(x,np.ones_like(y),c))
u1 = lambda x,y,c: gu(x,y,c)\
                  +vslow(np.zeros_like(x),y,c)-gu(np.zeros_like(x),y,c)
uslow = lambda x,y,c: u1(x,y,c)\
                  +y*(u1(x,np.zeros_like(y),c)-u1(x,np.ones_like(y),c))
v = jit(vslow)
u = jit(uslow)

# Plot the results
c = {'a':np.array([0.]),'b':np.array([0.])}
U = u(x,y,c).reshape((n,n)).T
V = v(x,y,c).reshape((n,n)).T

vmax = 3.2
vmin = 1.8

zero = np.zeros(n)
one = np.ones(n)
line = np.linspace(0.,1.,n)

c1 = u(zero,line,c)
c2 = u(line,zero,c)
c3 = v(line,zero,c)
c4 = v(one,line,c)

p = MP()
meshU = p.mesh(*dark,U,scalars=U,colormap='jet',vmax=vmax,vmin=vmin)
meshV = p.mesh(*darkV,V,scalars=V,colormap='jet',vmax=vmax,vmin=vmin)
l1 = p.plot3d(line,zero,c1,color='r',tube_radius=0.015)
l2 = p.plot3d(line+2.,zero,c1,color='r',tube_radius=0.015)
l3 = p.plot3d(zero,line,c2,color='k',tube_radius=0.015)
l4 = p.plot3d(one,line,c2,color='k',tube_radius=0.015)
l5 = p.plot3d(zero+2.,line,c3,color='k',tube_radius=0.015)
l6 = p.plot3d(one+2.,line,c3,color='k',tube_radius=0.015)
l7 = p.plot3d(line+2.,one,c4,color='m',tube_radius=0.015)
p.view(azimuth=-30,elevation=60,roll=-60)

a = np.linspace(0.,12.,150)
b = np.linspace(0.,0.3,150)
def anim():
    for k in range(300):
        if k < 150:
            c = {'a':a[k],'b':b[k]}
        if k >= 150:
            c = {'a':a[299-k],'b':b[299-k]}

        U = onp.array(u(x,y,c)).reshape((n,n)).T
        V = onp.array(v(x,y,c)).reshape((n,n)).T
        c1 = onp.array(u(zero,line,c))
        c2 = onp.array(u(line,zero,c))
        c3 = onp.array(v(line,zero,c))
        c4 = onp.array(v(one,line,c))

        meshU.mlab_source.z = U
        meshV.mlab_source.z = V
        meshU.mlab_source.scalars = U
        meshV.mlab_source.scalars = V
        l1.mlab_source.z = c1
        l2.mlab_source.z = c1
        l3.mlab_source.z = c2
        l4.mlab_source.z = c2
        l5.mlab_source.z = c3
        l6.mlab_source.z = c3
        l7.mlab_source.z = c4
        yield

p.show()
p.animate(anim,save=False)

print("After the plot displays, run: p.animate(anim,save=False)")
