import numpy as np

# Import the model from the auxillary folder
import sys
sys.path.append("aux")
from Example_3_4_aux import myModel

np.random.seed(1)

# Constants:
N = 10
Ntest = 100

H = 15
varType = "float64"

# Analytical solution
real = lambda x,y: np.exp(-x)*(x+y**3)

# Create the training data
X,Y = np.meshgrid(np.linspace(0,1,N),np.linspace(0,1,N))
X = X.flatten(); Y = Y.flatten()
inputs = np.vstack([X,Y])

inputs = np.array(inputs,dtype=varType).T

# Train the class
outputs = np.zeros((inputs.shape[0],1),dtype=varType)

model = myModel(H)
model.trainBfgs(inputs,outputs,maxIter=2000,tol=1e-14)

errUTrain = np.abs(model.call(inputs)-np.expand_dims(real(X,Y),1))

#: Print out the error statistics for the test set
Xtest, Ytest = np.meshgrid(np.linspace(0,1,Ntest),np.linspace(0,1,Ntest))
Xtest = Xtest.flatten(); Ytest = Ytest.flatten()

test = np.vstack([Xtest,Ytest]).T
test = np.array(test,dtype=varType)
u = model.call(test)
errU = np.abs(u-np.expand_dims(real(Xtest,Ytest),1))

print("")
print("Max Error U Train: "+str(np.max(errUTrain)))
print("Mean Error U Train: "+str(np.mean(errUTrain)))
print("Max Error U Test: "+str(np.max(errU)))
print("Mean Error U Test: "+str(np.mean(errU)))
