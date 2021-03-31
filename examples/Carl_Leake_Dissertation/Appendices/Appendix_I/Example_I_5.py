import jax.numpy as np
from jax import jit

from tfc.utils import egrad
from tfc.utils.PlotlyMakePlot import MakePlot

# Create X:
n = [50,50]
xMat = np.meshgrid(np.linspace(-4.,4.,50),np.linspace(-4.,4.,50))
x = xMat[0].flatten()+1.j*xMat[1].flatten()

# Create the constrained expression:
g = lambda x: 0.25*x + np.cos(x/4.)*0.3j

uslow = lambda x: g(x)\
                  +((-44.+8.j)*x**2+(52.+36.j)*x+(132.-24.j))/125.*(1.+np.pi*1.j-g(0.5j*np.ones_like(x)))\
                  +((69.+67.j)/125.*x**2-(129.+397.j)/250.*x+(-82.+49.j)/125.)*(g(np.ones_like(x))-g(1.j*np.ones_like(x)))\
                  +((44.-8.j)*x**2-(52.+36.j)*x+(-7.+24.j))/125.*(2.j-g(np.ones_like(x)*(2.+1.j))-egrad(g)(np.ones_like(x)))
u = jit(uslow)

# Plot:
p = MakePlot('Re[x]','Im[x]',zlabs='Re[u(x,g(x))]')
p2 = MakePlot('Re[x]','Im[x]',zlabs='Im[u(x,g(x))]')

# Add constraints
p.Scatter3d(x=[0.],
            y=[1./2.],
            z=[1.],
            mode="markers",
            marker=dict(color="red",size=5),
           )
p2.Scatter3d(x=[0.],
            y=[1./2.],
            z=[np.pi],
            mode="markers",
            marker=dict(color="red",size=5),
           )

U1 = u(np.array([1.,1.]))
p.Scatter3d(x=[1.,0.],
            y=[0.,1.],
            z=np.real(U1).tolist(),
            mode="markers",
            marker=dict(color="green",size=5),
           )
p2.Scatter3d(x=[1.,0.],
            y=[0.,1.],
            z=np.imag(U1).tolist(),
            mode="markers",
            marker=dict(color="green",size=5),
           )

# Add constrained expression
U = u(x)
p.Surface(x=xMat[0],y=xMat[1],z=np.real(U).reshape(n),showscale=False)
p2.Surface(x=xMat[0],y=xMat[1],z=np.imag(U).reshape(n),showscale=False)

# Change plot properties
p.fig.update_layout(showlegend=False,scene_aspectmode='cube')
p2.fig.update_layout(showlegend=False,scene_aspectmode='cube')
p.view(45,40)
p2.view(45,40)
p.show()
p2.show()

# Check the constraints:
e1 = 1.+np.pi*1.j-u(np.array([0.5j]))
e2 = u(np.array([1.+0.j]))-u(np.array([0.+1.j]))
e3 = 2.j-u(np.array([2.+1.j]))-egrad(u)(np.array([1.]))
