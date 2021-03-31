import numpy as onp
import jax.numpy as np
from jax import jit

from tfc.utils import egrad
from tfc.utils.PlotlyMakePlot import MakePlot

# Create X:
n = [20,20,20]
dark = np.linspace(0.,1.,n[0])
xMat = onp.mgrid[0:1:20j,0:1:20j,0:1:20j]
x = [k.flatten() for k in xMat]

# Create the constrained expression:
g = lambda *x: np.cos(x[0])*np.sqrt(x[1])+x[2]**2
u = lambda *x: g(*x)\
               +x[2]*(np.sin(x[0])*np.cos(x[1])-g(x[0],x[1],np.ones_like(x[2])))\
               +(1-x[2])*((1-x[0])*(np.exp(x[1])-g(np.zeros_like(x[0]),x[1],np.zeros_like(x[2])))\
               +x[0]*((1.-x[1])*(3.-g(np.ones_like(x[0]),np.zeros_like(x[1]),np.zeros_like(x[2])))\
               +x[1]*(5.-g(np.ones_like(x[0]),np.ones_like(x[1]),np.zeros_like(x[2])))))

# Plot:
p = MakePlot('x','y',zlabs='z')
U = u(*x)
p.Volume(x=x[0],y=x[1],z=x[2],value=U)
p.view(-45,30)
p.show()

# Error in constraints
e1 = np.max(np.abs(np.sin(x[0])*np.cos(x[1])-u(x[0],x[1],np.ones_like(x[2]))))
e2 = np.max(np.abs(np.exp(dark)-u(np.zeros_like(dark),dark,np.zeros_like(dark))))
e3 = np.max(np.abs(3.-u(np.array([1.]),np.array([0.]),np.array([0.]))))
e4 = np.max(np.abs(5.-u(np.array([1.]),np.array([1.]),np.array([0.]))))
