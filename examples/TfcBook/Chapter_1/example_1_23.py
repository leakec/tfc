import numpy as np

from tfc import utfc
from tfc.utils.TFCUtils import TFCPrint
from tfc.utils import MakePlot, step

# Problem constants:
N = 100
m = 10
nFunc = 7

# Quaternion constraints:
qi = np.array([np.cos(np.pi/3.),0.,np.sin(np.pi/3.),0.])
qf = np.array([1.,0.,0.,0.])

# Define the q2u and u2q transforms:
def q2u(q):
    if q.ndim == 1:
        num = 1
        flat = True
        q = np.expand_dims(q,1)
    else:
        num = q.shape[1]
        flat = False
    uvw = np.zeros((3,num))
    for k in range(num):
        norm = np.linalg.norm(q[:,k])
        if np.round(norm,15) != 1:
            TFCPrint.Error("Quaternion does not satisfy unit constraint!")
        uvw[0,k] = np.arccos(q[0,k])
        if np.all(q[1:,k] == 0):
            uvw[1,k] = 0.
        else:
            uvw[1,k] = np.arccos(q[1,k]/np.sqrt(np.sum(q[1:,k]**2)))
        if np.all(q[2:,k] == 0):
            uvw[2,k] = 0.
        else:
            if q[3,k] >= 0:
                uvw[2,k] = np.arccos(q[2,k]/np.sqrt(np.sum(q[2:,k]**2)))
            else:
                uvw[2,k] = 2.*np.pi-np.arccos(q[2,k]/np.sqrt(np.sum(q[2:,k]**2)))
        if flat:
            uvw = uvw.flatten()
    return uvw

def u2q(uvw):
    if uvw.ndim == 1:
        num = 1
        flat = True
        uvw = np.expand_dims(uvw,1)
    else:
        flat = False
        num = uvw.shape[1]
    q = np.zeros((4,num))
    for k in range(num):
        q[0,k] = np.cos(uvw[0,k])
        q[1,k] = np.sin(uvw[0,k])*np.cos(uvw[1,k])
        q[2,k] = np.sin(uvw[0,k])*np.sin(uvw[1,k])*np.cos(uvw[2,k])
        q[3,k] = np.sin(uvw[0,k])*np.sin(uvw[1,k])*np.sin(uvw[2,k])
    if flat:
        q = q.flatten()
    return q

# Generate the constraints in u,v,w:
uvwi = q2u(qi)
ui = uvwi[0]
vi = uvwi[1]
wi = uvwi[2]

uvwf = q2u(qf)
uf = uvwf[0]
vf = uvwf[1]
wf = uvwf[2]

# Create the TFC class:
myTfc = utfc(N,-1,m,x0=0.,xf=2.,basis='FS')
H = myTfc.H
t = myTfc.x

# Create the constrained expressions:
uhat = lambda t,g: g(t)\
                   +(2.-t)/2.*(ui-g(np.zeros_like(t)))\
                   +t/2.*(uf-g(2.*np.ones_like(t)))
vhat = lambda t,g: g(t)\
                   +(2.-t)/2.*(vi-g(np.zeros_like(t)))\
                   +t/2.*(vf-g(2.*np.ones_like(t)))
u = lambda t,g: uhat(t,g)\
                +(np.pi-uhat(t,g))*step(uhat(t,g)-np.pi)\
                -uhat(t,g)*step(-uhat(t,g))
v = lambda t,g: vhat(t,g)\
                +(np.pi-vhat(t,g))*step(vhat(t,g)-np.pi)\
                -vhat(t,g)*step(-vhat(t,g))
w = lambda t,g: np.mod(g(t),2.*np.pi)\
                   +(2.-t)/2.*(wi-np.mod(g(np.zeros_like(t)),2.*np.pi))\
                   +t/2.*(wf-np.mod(g(2.*np.ones_like(t)),2.*np.pi))

# Create the random paths and plot them
np.random.seed(2)

xlabs = np.array([r'$t$',]*4).reshape((2,2))
ylabs = np.array([r'$q_k$',]*4)
for k in range(4):
    myStr = r'$q_'+str(k)+r'$'
    ylabs[k] = myStr
ylabs = np.reshape(ylabs,(2,2))

p = MakePlot(xlabs,ylabs)
m = H(t).shape[1]
for k in range(nFunc):
    xiu = np.random.rand(m)-0.5
    xiv = np.random.rand(m)-0.5
    xiw = np.random.rand(m)-0.5
    gu = lambda t: np.dot(H(t),xiu)
    gv = lambda t: np.dot(H(t),xiv)
    gw = lambda t: np.dot(H(t),xiw)
    uvw = np.vstack([u(t,gu),v(t,gv),w(t,gw)])
    q = u2q(uvw)
    for j in range(4):
        p.ax[j].plot(t,q[j,:])

for j in range(4):
    p.ax[j].scatter(t[0],qi[j],color='k',s=30,zorder=21)
    p.ax[j].scatter(t[-1],qf[j],color='k',s=30,zorder=21)

p.fig.subplots_adjust(wspace=0.2)
p.FullScreen()
p.show()
