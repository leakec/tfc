import sys

from jax.config import config
config.update('jax_enable_x64', True)

import numpy as onp
import jax.numpy as np
import jax.nn as nn
import jax.random as random

from TFCUtils import TFCPrint, TFCDictRobust

nnActs = {'relu':nn.relu,
          'tanh':np.tanh,
          'hard_tanh':nn.hard_tanh,
          'sigmoid':nn.sigmoid,
          'swish':nn.swish}

initializers = {'xavier_uniform':nn.initializers.xavier_uniform(),
                'xavier_normal':nn.initializers.xavier_normal(),
                'normal':nn.initializers.normal,
                'uniform':nn.initializers.uniform,
                'zeros':nn.initializers.zeros}

class NN:

    def __init__(self,n,layers,
                 dimIn=2,dimOut=1,c=0.,x0=None,z=None,
                 act='tanh',weightInitializer='xavier_normal',biasInitializer='zeros',seed=None):

        self._dimIn = dimIn
        self._dimOut = dimOut

        # Set N based on user input
        if isinstance(n,np.ndarray): 
            if not n.flatten().shape[0] == dimIn:
                TFCPrint.Error("n has length "+str(n.flatten().shape[0])+", but it should be equal to the number of dimensions, "+str(dimIn)+".")
            self.n = n.astype(np.int32)
        else:
            if not len(n) == dimIn:
                TFCPrint.Error("n has length "+len(n)+", but it should be equal to the number of dimensions, "+str(dimIn)+".")
            self.n = np.array(n,dtype=np.int32)
        self.N = int(np.prod(self.n,dtype=np.int32))

        # Set x0 based on user input
        if x0 is None:
            self.x0 = np.zeros(dimIn)
        else:
            if isinstance(x0,np.ndarray):
                if not x0.flatten().shape[0] == dimIn:
                    TFCPrint.Error("x0 has length "+str(x0.flatten().shape[0])+", but it should be equal to the number of dimensions, "+str(dimIn)+".")
                self.x0 = x0
            else:
                if not len(x0) == dimIn:
                    TFCPrint.Error("x0 has length "+len(x0)+", but it should be equal to the number of dimensions, "+str(dimIn)+".")
                self.x0 = np.array(x0).flatten()
                if not x0.flatten().shape[0] == dimIn:
                    TFCPrint.Error("x0 has length "+str(x0.flatten().shape[0])+", but it should be equal to the number of dimensions, "+str(dimIn)+".")

        # Create c array based on user input
        if np.any(c==0):
            TFCPrint.Error("The value of c you have entered is invalid. Please enter a valid value for c.")
        if isinstance(c,np.ndarray):
            if not c.flatten().shape[0] == self._dimIn:
                TFCPrint.Error("c has length "+str(c.flatten().shape[0])+", but it should be equal to the number of dimensions, "+str(dimIn)+".")
            self.c = c.flatten()
        else:
            if not len(c) == self._dimIn:
                TFCPrint.Error("c has length "+len(c)+", but it should be equal to the number of dimensions, "+str(dimIn)+".")
            self.c = np.array(c)

        # Calculate z points and corresponding x
        if z is None:
            self.z = onp.zeros((self._dimIn,self.N))
            x = tuple([onp.zeros(self.N) for x in range(self._dimIn)])
            for k in range(self._dimIn):
                nProd = onp.prod(self.n[k+1:])
                nStack = onp.prod(self.n[0:k])
                dark = onp.linspace(0.,1.,num=self.n[k]).reshape((self.n[k],1))
                dark = onp.hstack([dark]*nProd).flatten()
                self.z[k,:] = onp.array([dark]*nStack).flatten()
                x[k][:] = self.z[k,:]/self.c[k] + self.x0[k]
        else:
            if not (z.shape[0] == self._dimIn and z.shape[1] == self.N):
                TFCPrint.Error("Input vector z is not the correct size. It is of size ("+str(z.shape[0])+","+str(self._dimIn)+"), but it should be size ("+str(self._dimIn)+","+str(self.N)+").")
            self.z = z
            x = tuple([onp.zeros(self.N) for x in range(self._dimIn)])
            for k in range(self._dimIn):
                x[k][:] = self.z[k,:]/self.c[k] + self.x0[k]

        self.z = np.array(self.z.tolist())
        self.x = tuple([np.array(x[k].tolist()) for k in range(self._dimIn)])

        # Set activation function based on user input
        if act in nnActs:
            self._act = nnActs[act]
        else:
            TFCPrint.Error("Activation function specified '"+act+"' is not one of the allowed activation functions.")

        # Set weight activation function based on user input
        if weightInitializer in initializers:
            self._wInit = initializers[weightInitializer]
        else:
            TFCPrint.Error("Weight initialization function specified '"+weightInitializer+"' is not one of the allowed weight initialization functions.")

        # Set bias activation function based on user input
        if biasInitializer in initializers:
            self._bInit = initializers[biasInitializer]
        else:
            TFCPrint.Error("Bias initialization function specified '"+biasInitializer+"' is not one of the allowed bias initialization functions.")

        # Set random prng key based on user seed input
        if seed is None:
            self._prngKey = random.PRNGKey(onp.random.randint(0,sys.maxsize))
        else:
            self._prngKey = random.PRNGKey(seed)

        # Initialize weights and biases
        self._xi = TFCDictRobust()

        if isinstance(layers,np.ndarray): 
            layers = layers.flatten()
        else:
            layers = np.array(layers,dtype=np.int32)
        self._nLayers = layers.shape[0]

        nBiases = np.sum(layers)+dimOut
        nWeights = dimIn*layers[0]+layers[-1]*dimOut
        for k in range(self._nLayers-1):
            nWeights += layers[k]*layers[k+1]

        weights = self._wInit(self._prngKey,(nWeights,1)).flatten()
        biases = self._bInit(self._prngKey,(nBiases,1))

        startW = 0
        stopW = dimIn*layers[0]
        dim1 = layers[0]
        dim2 = dimIn
        startB = 0
        stopB = layers[0]
        for k in range(self._nLayers):
            self._xi.update({'L'+str(k)+'w':weights[startW:stopW].reshape(dim1,dim2)})
            self._xi.update({'L'+str(k)+'b':biases[startB:stopB]})

            if not k == self._nLayers-1:
                startW = stopW
                stopW += layers[k]*layers[k+1]
                dim1 = layers[k+1]
                dim2 = layers[k]

                startB = stopB
                stopB += layers[k+1]
        self._xi.update({'L'+str(self._nLayers)+'w':weights[stopW:]})
        self._xi.update({'L'+str(self._nLayers)+'b':biases[stopB:]})
    
    def H(self):
        dim = self._dimIn
        nLayers = self._nLayers
        act = self._act
        def run(xi,*x):
            dark = np.outer(xi['L0w'][:,0],x[0])
            for k in range(dim-1):
                dark += np.outer(xi['L0w'][:,k+1],x[k+1])
            dark += xi['L0b']
            for k in range(nLayers):
                dark = np.dot(xi['L'+str(k+1)+'w'],act(dark))+xi['L'+str(k+1)+'b']
            return dark.flatten()
        return run
    
    @property
    def xi(self):
        return self._xi
    @xi.setter
    def xi(self,val):
        self._xi = val

    def RepMat(self,varIn,dim=1):
        """ This function is used to replicate a vector along the dimension specified by dim to create a matrix
            the same size as the H matrix."""
        if dim == 1:
            if not isinstance(self.nC,tuple):
                return np.tile(varIn,(1,self.deg+1-self.nC))
            else:
                return np.tile(varIn,(1,self.deg+1-self.nC.__len__()))
        elif dim == 0:
            return np.tile(varIn,(self.N,1))
        else:
            TFCPrint.Error('Invalid dimension')
