import os,sys
sys.path.append(os.path.join('..','..','..','src','build','bin'))
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from time import perf_counter as timer
import numpy as np
import numpy.matlib
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import cm

from DeepTFCBurgers1DClass import myModel
from MakePlot import MakePlot

# Set warning level for tensorflow
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(1)

# Constants:
N = 900
Nuniform = 100

Ntest = 100
H = 30

c = 1.
alpha = 1.
nu = 0.5
x0 = -3.
xf = 3.

varType = "float64"

# Create the training data
X,T = np.meshgrid(np.linspace(x0,xf,int(np.sqrt(N))),np.linspace(0,1,int(np.sqrt(N))))
X = X.flatten(); T = T.flatten()
inputs = np.vstack([X,T])

#xU,tU = np.meshgrid(np.linspace(x0,xf,int(np.sqrt(Nuniform))),\
#                    np.linspace(0.,1.,int(np.sqrt(Nuniform))))
#xU = xU.flatten(); tU = tU.flatten()
#
#inputs = np.zeros((2,N))
#count = xU.shape[0]
#inputs[0,:count] = xU; inputs[1,:count] = tU
#
#f = lambda x,t: 1./np.cosh(c*(-c*t+x)/(2.*nu))
#while count < N:
#    x = np.random.rand()*(xf-x0)+x0
#    t = np.random.rand()
#    u = np.random.rand()
#    if u < f(x,t):
#        inputs[:,count] = np.array([x,t])
#        count += 1

inputs = np.array(inputs,dtype=varType).T

# Train the class
model = myModel(c,nu,alpha,x0,xf,H)
outputs = np.zeros((inputs.shape[0],1),dtype=varType)

start = timer()
model.trainBfgs(inputs,outputs,maxIter=1500)
end = timer()

#: Print out the error statistics for the test set
Xtest, Ttest = np.meshgrid(np.linspace(x0,xf,Ntest),np.linspace(0,1,Ntest))
Xtest = Xtest.flatten(); Ttest = Ttest.flatten()

test = np.vstack([Xtest,Ttest]).T
test = np.array(test,dtype=varType)
u = model.call(test)
errU = np.abs(u-np.expand_dims(model.ua(Xtest,Ttest),1))

print("")
print("Training time: "+str(end-start))
print("Max Error U: "+str(np.max(errU)))
print("Mean Error U: "+str(np.mean(errU)))

#p = MakePlot([['x',]*2],[['t',]*2],zlabs=[['u','err']])
#p.ax[0].plot_surface(Xtest.reshape((Ntest,Ntest)),Ttest.reshape((Ntest,Ntest)),u.numpy().reshape((Ntest,Ntest)))
#p.ax[1].plot_surface(Xtest.reshape((Ntest,Ntest)),Ttest.reshape((Ntest,Ntest)),errU.reshape((Ntest,Ntest)))
#p.show()

