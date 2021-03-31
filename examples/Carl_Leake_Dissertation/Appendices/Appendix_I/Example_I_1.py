import jax.numpy as np
from jax import jit

from tfc.utils import egrad
from tfc.utils.PlotlyMakePlot import MakePlot

# Create X:
n = [50,50]
dark = np.meshgrid(np.linspace(0.,1.,50),np.linspace(0.,1.,50))
X = [k.flatten() for k in dark]

# Create the affine transformation and its inverse:
p0 = np.array([2.,3.]).reshape((2,1))
p1 = np.array([4.,4.]).reshape((2,1))
p2 = np.array([3.,4.]).reshape((2,1))
a = np.hstack([p1-p0,p2-p0])
ainv = np.linalg.inv(a)

def A(*X):
    dark = np.dot(a,np.vstack([*X]))+p0
    return (dark[0,:],dark[1,:])

def Ainv(*x):
    dark = np.dot(ainv,np.vstack([*x])-p0)
    return (dark[0,:],dark[1,:])

# Create the constrained expression:
n1 = np.array([2.,1.])
n1mag = np.linalg.norm(n1)
n1 = n1/n1mag

n2 = np.array([1.,1.])
n2mag = np.linalg.norm(n2)

g = lambda xi,*x: np.sin(x[0])*np.cos(x[1])
dgn1 = lambda xi,*x: egrad(g,1)(xi,*x)*n1[0]+egrad(g,2)(xi,*x)*n1[1]
intgn2 = lambda t: -1./2.*np.cos(t)**2*np.sqrt(2.)

u1 = lambda xi,*x: g(xi,*x)+\
                   n1mag*(Ainv(*x)[0]-1.)*(0.5-dgn1(xi,*A(np.zeros_like(x[0]),Ainv(*x)[1])))+\
                   1./n2mag*(-2.-intgn2(5.)+intgn2(4.))
uslow = lambda xi,*x: u1(xi,*x)+\
                  (1./2.-Ainv(*x)[1])*(u1(xi,*A(Ainv(*x)[0],np.ones_like(x[1])))-u1(xi,*A(Ainv(*x)[0],np.zeros_like(x[1]))))
u = jit(uslow)

# Plot:
x = A(*X)
xi = np.zeros(5)
xmat = [k.reshape(n) for k in x]

p = MakePlot('x','y',zlabs='u(x,y,g(x,y))')

# Add constraints
c3x = (np.linspace(2.,4.,100),np.linspace(3.,4.,100))
c3u = u(xi,*c3x)
p.Scatter3d(x=c3x[0],
          y=c3x[1],
          z=c3u,
          mode="lines",
          line=dict(color="red",width=5)
          )
p.Scatter3d(x=c3x[0]+1.,
          y=c3x[1]+1.,
          z=c3u,
          mode="lines",
          line=dict(color="red",width=5)
          )

c2x = (np.linspace(2.,3.,100),np.linspace(3.,4.,100))
c2u = u(xi,*c2x)
for k in range(c2x[0].shape[0]):
    if not k%5:
        p.Scatter3d(x=np.array([c2x[0][k],c2x[0][k]+n1[0]/15.]),
                    y=np.array([c2x[1][k],c2x[1][k]+n1[1]/15.]),
                    z=np.array([c2u[k],c2u[k]+0.5/15.]),
                    mode='lines',
                    line=dict(color='green',width=5))

# Add constrained expression
Z = u(xi,*x)
p.Surface(x=xmat[0],y=xmat[1],z=Z.reshape(n),showscale=False,
          contours_z=dict(show=True, usecolormap=True,
                          size=0.1, width=1, project_z=True)
          )

# Add plot of parallelotope
arr = np.hstack([p0,p1,p1+p2-p0,p2,p0])
p.Scatter3d(x=arr[0,:],
            y=arr[1,:],
            z=(Z.min()-0.05)*np.ones_like(arr[0,:]),
            mode="lines",
            line=dict(color="black",width=8,dash='solid')
            )

# Change plot properties
p.fig.layout.scene.zaxis.range = [(Z.min()-0.06),Z.max()+0.06]
p.fig.update_layout(showlegend=False,scene_aspectmode='cube')
p.view(-45,30)
p.show()
