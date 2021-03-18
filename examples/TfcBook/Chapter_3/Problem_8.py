import jax.numpy as np
from tfc import mtfc
from tfc.utils import egrad, NLLS
from tfc.utils.PlotlyMakePlot import MakePlot

# Constants:
n = [40,40]
nC = [2,[1,2]]
m = 40

r0 = 2.
rf = 4.
th0 = 0.
thf = 2.*np.pi

realSoln = lambda r,th: 4.*(-1024.+r**10)*np.sin(5.*th)/(1023.*r**5)

# Create TFC class:
myTfc = mtfc(n,nC,m,x0=[r0,th0],xf=[rf,thf])
H = myTfc.H
x = myTfc.x

# Create constrained expression:
g = lambda xi,*x: np.dot(H(*x),xi)
u1 = lambda xi,*x: g(xi,*x)+\
                   (x[0]-rf)/(r0-rf)*(0.-g(xi,r0*np.ones_like(x[0]),x[1]))+\
                   (x[0]-r0)/(rf-r0)*(4.*np.sin(5.*x[1])-g(xi,rf*np.ones_like(x[0]),x[1]))
u = lambda xi,*x: u1(xi,*x)+\
                  -x[1]/(2.*np.pi)*(u1(xi,x[0],thf*np.ones_like(x[1]))-u1(xi,x[0],th0*np.ones_like(x[1])))+\
                  (-x[1]**2+2.*np.pi*x[1])/(4.*np.pi)*(egrad(u1,2)(xi,x[0],thf*np.ones_like(x[1]))-egrad(u1,2)(xi,x[0],th0*np.ones_like(x[1])))
                  

# Create the loss function:
ur = egrad(u,1)
u2r = egrad(ur,1)
u2th = egrad(egrad(u,2),2)
L = lambda xi: u2r(xi,*x)+1./x[0]*ur(xi,*x)+1./x[0]**2*u2th(xi,*x)

# Solve the problem:
xi = np.zeros(H(*x).shape[1])
xi,it,time = NLLS(xi,L,timer=True)

# Print out statistics:
print("Solution time: {0} seconds".format(time))

# Plot the solution:
R,Th = np.meshgrid(np.linspace(r0,rf,50),np.linspace(th0,thf,200))
dark = (R.flatten(),Th.flatten())

X = R*np.cos(Th)
Y = R*np.sin(Th)
U = u(xi,*dark).reshape((200,50))
p = MakePlot("x","y",zlabs="u(x,y,g(x,y))")
p.Surface(x=X,y=Y,z=U,showscale=False)
p.show()

# Plot the error
err = np.abs(realSoln(R,Th)-U)
p = MakePlot("x","y",zlabs="Error")
p.Surface(x=X,y=Y,z=err,showscale=False)
p.show()
