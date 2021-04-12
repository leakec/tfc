import jax.numpy as np
from tfc.utils.PlotlyMakePlot import MakePlot

# Create X:
n = [40,40]
dark = np.linspace(-1.,1.,n[0])
xMat = np.meshgrid(dark,dark)
x = [k.flatten() for k in xMat]

# Create the constrained expression:
g = lambda *x: np.sin(3.*np.pi/4.*x[0])*np.sin(np.pi/2.*x[1])
u = lambda *x: g(*x)+5.-g(np.zeros_like(x[0]),np.zeros_like(x[1]))

# Plot:
p = MakePlot('x','y',zlabs='u(x,y,g(x,y))')
U = u(*x)
p.Surface(x=xMat[0],y=xMat[1],z=U.reshape(n),showscale=False)
p.Scatter3d(x=[0.],y=[0.],z=[5.],mode='markers',
            marker=dict(size=4),line=dict(color='red'))
p.view(-45,50)
p.show()
