import numpy as np
from tfc.utils.PlotlyMakePlot import MakePlot

# Coon's patch functions
A = lambda x,y: np.array([[0.,3.*x**2,3.*x*np.cos(2.*np.pi*x)],
                          [np.sin(2.*np.pi*y), 0., 0.],
                          [3.*np.cos(4.*np.pi*y),-3.,-3.]])

# Constrained expression functions:
g = lambda x,y: 3./5.*np.sin(6.*np.pi*x)*np.cos(5.*np.pi*y)*np.exp(x)
M = lambda x,y: np.array([[0.,3.*x**2-g(x,0.),3.*x*np.cos(2.*np.pi*x)-g(x,1.)],
                          [np.sin(2.*np.pi*y)-g(0.,y),g(0.,0.),g(0.,1.)],
                          [3.*np.cos(4.*np.pi*y),g(1.,0.)-3.,g(1.,1.)-3.]])
PhiX = lambda x: np.array([1., 1.-x, x])
PhiY = lambda y: np.array([1., 1.-y, y])

# Create grid
x,y = np.meshgrid(np.linspace(0.,1.,100),np.linspace(0.,1.,100))

# Create plot data
ce = np.zeros_like(x)
coons = np.zeros_like(x)
for j in range(x.shape[0]):
    for k in range(x.shape[1]):
        phix = PhiX(x[j,k])
        phiy = PhiX(y[j,k])
        coons[j,k] = np.linalg.multi_dot([phix,A(x[j,k],y[j,k]),phiy])
        ce[j,k] = g(x[j,k],y[j,k]) + np.linalg.multi_dot([phix,M(x[j,k],y[j,k]),phiy])

# Create the plots
line = np.linspace(0.,1.,100)
zero = np.zeros_like(line)
one = np.ones_like(line)
width = 7

p1 = MakePlot('x','y',zlabs='u(x,y,g(x,y))')
p1.Surface(x=x,y=y,z=ce,showscale=False)
p1.Scatter3d(x=zero,y=line,z=np.sin(2.*np.pi*line),
             mode='lines',
             line=dict(color='red',width=width))
p1.Scatter3d(x=one,y=line,z=3.*np.cos(4.*np.pi*line),
             mode='lines',
             line=dict(color='red',width=width))
p1.Scatter3d(x=line,y=zero,z=3.*line**2,
             mode='lines',
             line=dict(color='red',width=width))
p1.Scatter3d(x=line,y=one,z=3.*line*np.cos(2.*np.pi*line),
             mode='lines',
             line=dict(color='red',width=width))
p1.view(45,30)
p1.fig['layout']['showlegend'] = False
p1.show()

p2 = MakePlot('x','y',zlabs='u(x,y,0)')
p2.Surface(x=x,y=y,z=coons,showscale=False)
p2.Scatter3d(x=zero,y=line,z=np.sin(2.*np.pi*line),
             mode='lines',
             line=dict(color='red',width=width))
p2.Scatter3d(x=one,y=line,z=3.*np.cos(4.*np.pi*line),
             mode='lines',
             line=dict(color='red',width=width))
p2.Scatter3d(x=line,y=zero,z=3.*line**2,
             mode='lines',
             line=dict(color='red',width=width))
p2.Scatter3d(x=line,y=one,z=3.*line*np.cos(2.*np.pi*line),
             mode='lines',
             line=dict(color='red',width=width))
p2.view(45,30)
p2.fig['layout']['showlegend'] = False
p2.show()
