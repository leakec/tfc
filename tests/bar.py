from numpy import random
random.seed(0)

import jax.numpy as np
from tfc import utfc
from tfc.utils import egrad, NLLS
from tfc.utils.PlotlyMakePlot import MakePlot

# Domain circle of radius 1 around 1+1.j
# Problem: y''+ y*y' = f2(x)
# s.t. 
# Real solution: exp(-t) sin(t)

# Constants used in the differential equation:
f2 = lambda x: np.exp(-2.*x)*np.sin(x)*(np.cos(x)-np.sin(x))-2.*np.exp(-x)*np.cos(x)

# Store the real solution
realSoln = lambda x: np.exp(-x)*np.sin(x)

# Create UTFC class
a = np.sin(np.pi/4.)
x0 = 1-a + (1-a)*1.j
xf = 1+a + (1+a)*1.j
tfc = utfc(100, 0, 95, basis="ELMTanh", x0=x0, xf=xf, backend="Python")
H = tfc.H

# Set weigths and biases
size = tfc.basisClass.b.size

r = random.uniform(low=0.0, high=1.0, size=size)
th = random.uniform(low=0.0, high=2.0 * np.pi, size=size)
tfc.basisClass.w = r*(np.cos(th)+np.sin(th)*1.j)

r = random.uniform(low=0.0, high=1.0, size=size)
th = random.uniform(low=0.0, high=2.0 * np.pi, size=size)
tfc.basisClass.b = r*(np.cos(th)+np.sin(th)*1.j)

# Create the points 
r = np.linspace(0.,1.,10).reshape((1,10))
th = np.linspace(0.,2*np.pi,10).reshape((10,1))
real = r*np.sin(th)
imag = r*np.cos(th)*1.j
x = (real+imag).flatten() + 1. + 1.j

# Create constrained expression
g = lambda x,xi: np.dot(H(x),xi)
u = lambda x,xi: g(x,xi) + (x-1.0)/(0.2 + 1.2j)*(np.exp(-1.2 -1.2j)*np.sin(1.2+1.2j)-g(np.ones_like(x)*(1.2 + 1.2j), xi)) + (x-1.2 - 1.2j)/(-0.2 -1.2j)*(np.exp(-1.0)*np.sin(1.0) - g(np.ones_like(x),xi))

# Create loss function
ud = egrad(u)
udd = egrad(ud)
L = lambda xi,x: udd(x,xi) + ud(x,xi) * u(x,xi) - f2(x)

# Solve the problem
xi = np.zeros(H(x).shape[1], dtype=x.dtype)
xi, _, time = NLLS(xi,L,x,constant_arg_nums=[1], method="lstsq", timer=True, holomorphic=True, tol=1e-9)

# Create test points
numTest = 100
r = np.linspace(0.,1.,numTest).reshape((1,numTest))
th = np.linspace(0.,2*np.pi,numTest).reshape((numTest,1))
real = r*np.sin(th)
imag = r*np.cos(th)*1.j
test = (real+imag).flatten() + 1. + 1.j

# Calculate the error
U = u(test,xi)
err = U - realSoln(test)
maxErr = np.max(np.abs(err))

# Display error statistics
print(f"Time: {time}")
print(f"Max error: {maxErr}")

# Show the results
test = test.reshape((numTest, numTest))
U = U.reshape((numTest, numTest))
p = MakePlot([["x<sub>real</sub>","x<sub>real</sub>"]],[["x<sub>imag</sub>","x<sub>imag</sub>"]],zlabs=[["u<sub>real</sub>","u<sub>imag</sub>"]])

p.Surface(x=np.real(test), y=np.imag(test), z=np.real(U), row=1, col=1, showscale=False)
p.Surface(x=np.real(test), y=np.imag(test), z=np.imag(U), row=1, col=2, showscale=False)

p.Scatter3d(x=[1., 1.2],
            y=[0., 1.2],
            z=[np.exp(-1.0)*np.sin(1.0), np.real(np.exp(-1.2 - 1.2j)*np.sin(1.2j + 1.2))],
            mode="markers",
            marker=dict(color="red",size=5),
            row = 1, col = 1
           )
p.Scatter3d(x=[1., 1.2],
            y=[0., 1.2],
            z=[0.0, np.imag(np.exp(-1.2 -1.2j)*np.sin(1.2 + 1.2j))],
            mode="markers",
            marker=dict(color="red",size=5),
            row = 1, col = 2
           )

p.fig.update_layout(showlegend=False,scene_aspectmode='cube')
p.view(azimuth=225,elevation=25, row=1, col=1)
p.view(azimuth=225,elevation=25, row=1, col=2)
p.FullScreen()
p.show()
