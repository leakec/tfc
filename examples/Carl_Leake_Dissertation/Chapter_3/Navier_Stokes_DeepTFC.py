import numpy as np
import numpy.matlib
from matplotlib import cm

from tfc.utils import MakePlot

# Import the model from the auxillary folder
import sys
sys.path.append("aux")
from Navier_Stokes_DeepTFC_aux import myModel

# Set CPU as available physical device
#import tensorflow as tf
#my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
#tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

# Constants:
H = 30
varType = "float64"

L = 15.
tf = 3.
rho = 1.
mu = 1.
h = 1.
P = -5.

# Train the class
model = myModel(rho,h,L,P,mu,H)
inputs = np.random.rand(2000,3)*np.array([[L,h,tf]])+np.array([[0.,-h/2.,0.]])
inputs = np.array(inputs,dtype=varType)
outputs = np.zeros((inputs.shape[0],1),dtype=varType)
#model.trainBfgs(inputs,outputs,maxIter=1500)
model.trainBfgs(inputs,outputs,maxIter=500)

# Calcualte u and v and plot for different times
n = 100
X = np.matlib.repmat(np.reshape(np.linspace(0,L,num=n),(n,1)),n,1).flatten()
Y = np.reshape(np.matlib.repmat(np.reshape(np.linspace(-h/2.,h/2.,num=n),(n,1)),1,n),(n**2,1)).flatten()
xTest = np.zeros((3,n**2*3))
xTest[0,:] = np.hstack([X,]*3)
xTest[1,:] = np.hstack([Y,]*3)
xTest[2,:] = np.hstack([np.ones(n**2)*0.01,np.ones(n**2)*0.1,np.ones(n**2)*tf])
xTest = np.array(xTest.T,dtype=varType)

p = []; U = [];
vals = [0.01,0.1,tf]
u,v = model.call(xTest)
u = u.numpy(); v = v.numpy()
for k in range(len(vals)):
    p.append(MakePlot(r'$x (m)$',r'$y (m)$'))
    ind = np.where(np.round(xTest[:,2],12)==np.round(vals[k],12))
    U.append(np.reshape(u[ind],(n,n)))

Xm = np.reshape(xTest[:,0][ind],(n,n))
Ym = np.reshape(xTest[:,1][ind],(n,n))

dark = np.block(U)
vMin = np.min(dark)
vMax = np.max(dark)
def MakeContourPlot(Xm,Ym,Um):
    p = MakePlot(r'$x$ (m)',r'$y$ (m)')
    C = p.ax[0].contourf(Xm,Ym,Um,vmin=vMin,vmax=vMax,cmap=cm.gist_rainbow)
    cbar = p.fig.colorbar(C)
    return p

plots = [MakeContourPlot(Xm,Ym,U[0]),MakeContourPlot(Xm,Ym,U[1]),MakeContourPlot(Xm,Ym,U[2])]
for k,j in enumerate(plots):
    j.FullScreen()
    j.show()
    j.save('DeepTFC'+str(k),fileType='png')

# U error
ind = np.where(xTest[:,2]==tf)
ind2 = np.where(xTest[:,0][ind] == L)
uEnd = u[ind][ind2].flatten()
y = xTest[:,1][ind][ind2]
uTrue = P*(4.*y**2-h**2)/(8.*mu)
uErr = np.abs(uEnd-uTrue)
print("Max u error at the end: "+str(np.max(uErr)))
print("Mean u error at the end: "+str(np.mean(uErr)))

# V error
vEnd = v[ind][ind2].flatten()
vTrue = np.zeros_like(vEnd)
vErr = np.abs(vEnd-vTrue)
print("Max v error at the end: "+str(np.max(vErr)))
print("Mean v error at the end: "+str(np.mean(vErr)))
