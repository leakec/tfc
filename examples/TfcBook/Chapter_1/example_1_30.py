import jax.numpy as np

from tfc.utils import egrad
from tfc.utils.PlotlyMakePlot import MakePlot

SAVE_PLOTS = True
SHOW_PLOTS = False

# Constants:
n = 300

# Create domain:
x = np.linspace(-np.pi-0.5,np.pi+0.5,n)
y = np.linspace(-2.5,3.5,n)
X = np.meshgrid(x,y)

# Create constrained expression:
def u(g,*X):
    x = X[0]
    y = X[1]
    return g(*X)+\
           (y**2-1.)/3.*(np.sin(2.*x)-g(x,-2.*np.ones_like(y)))+\
           -(y**2+y-2.)*egrad(g,1)(x,np.zeros_like(y))+\
           (4.-y**2)/3.*(9.*np.exp(-x**2)-g(x,np.ones_like(y)))

# Create plot 1:
g = lambda *x: np.zeros_like(x[0])

lineKwargs = dict(mode='lines',showlegend=False,
             line=dict(width=5,color='black'))
zLift = 0.05

p = MakePlot('x','y',zlabs='u(x,y,g(x,y))')
p.Scatter3d(x=x,y=-2.*np.ones_like(y),z=np.sin(2.*x)+zLift,**lineKwargs)
p.Scatter3d(x=x,y=np.ones_like(y),z=9.*np.exp(-x**2)+zLift,**lineKwargs)
p.Surface(x=X[0],y=X[1],z=u(g,*X),colorscale='hsv',showscale=False)
p.view(-135.,25.)

# Create plot 2:
g = lambda x,y: 3.*x**2*y-2.*np.sin(15.*x)*np.cos(2.*y)
p1 = MakePlot('x','y',zlabs='u(x,y,g(x,y))')
p1.Scatter3d(x=x,y=-2.*np.ones_like(y),z=np.sin(2.*x)+zLift,**lineKwargs)
p1.Scatter3d(x=x,y=np.ones_like(y),z=9.*np.exp(-x**2)+zLift,**lineKwargs)
p1.Surface(x=X[0],y=X[1],z=u(g,*X),colorscale='hsv',showscale=False)
p1.view(-135.,25.)

# Show and save plots
if SHOW_PLOTS:
    p.show()
    p1.show()
if SAVE_PLOTS:
    p.save('1VariableEx2_1')
    p1.save('1VariableEx2_2')

