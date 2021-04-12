import numpy as onp
import jax.numpy as np
from jax import jit
from plotly.colors import qualitative as qual

from tfc import utfc
from tfc.utils import step, TFCDict
from tfc.utils.PlotlyMakePlot import MakePlot

# Constants:
aHyp = 0.1
bHyp = 0.1
cHyp = 0.3

aEll = 1.
bEll = 0.75
cEll = 2.

n = 100
nC = 2
m = 3
x0 = 0.
xf = 3.

nCEs = 8

colors = qual.Plotly

# Constrained expressions:
myTfc = utfc(n,nC,m,x0=x0,xf=xf)
t = myTfc.x
H = myTfc.H

g = lambda xi,t: np.dot(H(t),xi['xi'])
xslow = lambda xi,t: g(xi,t)\
                 +(3.-t)/3.*(aEll*np.sin(xi['phi'])*np.cos(xi['th'])-g(xi,np.array([0.])))\
                 +t/3.*(3.+aHyp*np.sinh(xi['v'])*np.cos(xi['psi'])-g(xi,np.array([3.])))
yslow = lambda xi,t: g(xi,t)\
                 +(3.-t)/3.*(bEll*np.sin(xi['phi'])*np.sin(xi['th'])-g(xi,np.array([0.])))\
                 +t/3.*(bHyp*np.sinh(xi['v'])*np.sin(xi['psi'])-g(xi,np.array([3.])))
zslow = lambda xi,t: g(xi,t)\
                 +(3.-t)/3.*(cEll*np.cos(xi['phi'])-g(xi,np.array([0.])))\
                 +t/3.*((-1.)**step(xi['n'])*cHyp*np.cosh(xi['v'])-g(xi,np.array([3.])))

x = jit(xslow); y = jit(yslow); z = jit(zslow)

# Plot conic sections
th = np.linspace(0.,2.*np.pi,100)
v = np.linspace(0.,3.0,100)
matHyp = np.meshgrid(v,th)

phi = np.linspace(0.,np.pi,100)
matEll = np.meshgrid(phi,th)

xHyp = lambda n,v,th: aHyp*np.sinh(np.abs(v))*np.cos(th)+3.
yHyp = lambda n,v,th: bHyp*np.sinh(np.abs(v))*np.sin(th)
zHyp = lambda n,v,th: (-1.)**step(n)*cHyp*np.cosh(np.abs(v))

xEll = lambda phi,th: aEll*np.sin(phi)*np.cos(th)
yEll = lambda phi,th: bEll*np.sin(phi)*np.sin(th)
zEll = lambda phi,th: cEll*np.cos(phi)

opacity = 0.6

p = MakePlot('x','y',zlabs='z')
p.Surface(x=xHyp(1.,*matHyp),
          y=yHyp(1.,*matHyp),
          z=zHyp(1.,*matHyp),
          showscale=False,
          colorscale=[[0.,"rebeccapurple"],[1.,"rebeccapurple"]],
          opacity=opacity)
p.Surface(x=xHyp(-1.,*matHyp),
          y=yHyp(-1.,*matHyp),
          z=zHyp(-1.,*matHyp),
          showscale=False,
          colorscale=[[0.,"rebeccapurple"],[1.,"rebeccapurple"]],
          opacity=opacity)
p.Surface(x=xEll(*matEll),
          y=yEll(*matEll),
          z=zEll(*matEll),
          showscale=False,
          colorscale=[[0.,"black"],[1.,"black"]],
          opacity=opacity)

# Plot the constrained expressions
onp.random.seed(1)
ind = np.array([0,-1])
m = H(t).shape[1]
for k in range(nCEs):
    xi = TFCDict({'xi':onp.random.randn(m)/3.,
                  'psi':onp.random.randn(1),
                  'phi':onp.random.randn(1),
                  'v':onp.random.rand(1)*6.-3.,
                  'th':onp.random.randn(1),
                  'n':onp.random.randn(1)})
    X = x(xi,t)
    Y = y(xi,t)
    Z = z(xi,t)
    p.Scatter3d(x=X,y=Y,z=Z,
                mode="lines",line=dict(color=colors[k],width=3))
    p.Scatter3d(x=X[ind],y=Y[ind],z=Z[ind],
                mode="markers",
                marker=dict(size=4),
                line=dict(color=colors[k]))

# Set plot properties and display it
p.fig.update_layout(showlegend=False,scene_aspectmode='cube')
p.view(-45,20)
p.show()

