import os,sys
sourcePath = os.path.join("..","src","build","bin")
sys.path.append(sourcePath)

from jax.config import config
config.update('jax_enable_x64', True)
import numpy as onp
import jax.numpy as np
from jax import vmap, grad

from BF import nCP, nLeP, nFS, nELMSigmoid, nELMTanh, nELMSin, nELMSwish
from TFCUtils import egrad

def test_nCP():
    from pOP import nCP as pnCP
    dim = 2
    nC = -1.*np.ones((dim,1),dtype=np.int32)
    d = np.zeros(dim,dtype=np.int32)
    c = np.ones(dim)
    d2 = np.array([2,3],dtype=np.int32)
    nC2Py = np.array([4,7],dtype=np.int32)
    nC2 = np.block([[np.arange(4),-1.*np.ones(3)],[np.arange(7)]]).astype(np.int32)
    n = np.array([10]*dim)
    N = np.prod(n)
    z = np.linspace(0,5,num=n[0])
    x = onp.zeros((N,dim))
    for k in range(dim):
        nProd = np.prod(n[k+1:])
        nStack = np.prod(n[0:k])
        dark = np.hstack([z]*nProd)
        x[:,k] = onp.array([dark]*nStack).flatten()

    useVal = np.array([0,0],dtype=np.int32)
    ncp1 = nCP(x.T,nC,5,c)
    ncp2 = nCP(x.T,nC2,10,c)
    Fc1 = ncp1.H(x.T,d,False,useVal)
    Fc2 = ncp2.H(x.T,d2,False,useVal)

    Fp1 = pnCP(x,4,d,nC.flatten()*0.)
    Fp2 = pnCP(x,9,d2,nC2Py)

    assert(np.linalg.norm(Fc1-Fp1,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc2-Fp2,ord='fro') < 1e-14)

def test_nLeP():
    from pOP import nLeP as pnLeP
    dim = 2
    nC = -1.*np.ones((dim,1),dtype=np.int32)
    d = np.zeros(dim,dtype=np.int32)
    c = np.ones(dim)
    d2 = np.array([2,3],dtype=np.int32)
    nC2Py = np.array([4,7],dtype=np.int32)
    nC2 = np.block([[np.arange(4),-1.*np.ones(3)],[np.arange(7)]]).astype(np.int32)
    n = np.array([10]*dim)
    N = np.prod(n)
    z = np.linspace(0,5,num=n[0])
    x = onp.zeros((N,dim))
    for k in range(dim):
        nProd = np.prod(n[k+1:])
        nStack = np.prod(n[0:k])
        dark = np.hstack([z]*nProd)
        x[:,k] = onp.array([dark]*nStack).flatten()

    useVal = np.array([0,0],dtype=np.int32)
    nlep1= nLeP(x.T,nC,5,c)
    nlep2= nLeP(x.T,nC2,10,c)
    Fc1 = nlep1.H(x.T,d,False,useVal)
    Fc2 = nlep2.H(x.T,d2,False,useVal)

    Fp1 = pnLeP(x,4,d,nC.flatten()*0.)
    Fp2 = pnLeP(x,9,d2,nC2Py)

    assert(np.linalg.norm(Fc1-Fp1,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc2-Fp2,ord='fro') < 1e-14)

def test_nFS():
    from pOP import nFS as pnFS
    dim = 2
    nC = -1.*np.ones((dim,1),dtype=np.int32)
    d = np.zeros(dim,dtype=np.int32)
    c = np.ones(dim)
    d2 = np.array([2,3],dtype=np.int32)
    nC2Py = np.array([4,7],dtype=np.int32)
    nC2 = np.block([[np.arange(4),-1.*np.ones(3)],[np.arange(7)]]).astype(np.int32)
    n = np.array([10]*dim)
    N = np.prod(n)
    z = np.linspace(0,5,num=n[0])
    x = onp.zeros((N,dim))
    for k in range(dim):
        nProd = np.prod(n[k+1:])
        nStack = np.prod(n[0:k])
        dark = np.hstack([z]*nProd)
        x[:,k] = onp.array([dark]*nStack).flatten()

    useVal = np.array([0,0],dtype=np.int32)
    nfs1 = nFS(x.T,nC,5,c)
    nfs2 = nFS(x.T,nC2,10,c)
    Fc1 = nfs1.H(x.T,d,False,useVal)
    Fc2 = nfs2.H(x.T,d2,False,useVal)

    Fp1 = pnFS(x,4,d,nC.flatten()*0.)
    Fp2 = pnFS(x,9,d2,nC2Py)

    assert(np.linalg.norm(Fc1-Fp1,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc2-Fp2,ord='fro') < 1e-14)

def test_nELMSigmoid():
    from pOP import nLeP as pnLeP
    dim = 2
    nC = -1*np.ones(1,dtype=np.int32)
    d = np.zeros(dim,dtype=np.int32)
    c = np.ones(dim)
    d2 = np.array([2,3],dtype=np.int32)
    nC2 = np.array([4],dtype=np.int32)
    n = np.array([10]*dim)
    N = np.prod(n)
    z = np.linspace(0,5,num=n[0])
    X = onp.zeros((N,dim))
    for k in range(dim):
        nProd = np.prod(n[k+1:])
        nStack = np.prod(n[0:k])
        dark = np.hstack([z]*nProd)
        X[:,k] = onp.array([dark]*nStack).flatten()

    useVal = np.array([0,0],dtype=np.int32)
    elm1 = nELMSigmoid(X.T,nC,10,c)
    w = elm1.w
    b = elm1.b
    elm2 = nELMSigmoid(X.T,nC2,10,c)
    elm2.w = w
    elm2.b = b
    Fc1 = elm1.H(X.T,d,False,useVal)
    Fc2 = elm2.H(X.T,d2,False,useVal)

    x = np.ones((100,10))*X[:,0:1]
    y = np.ones((100,10))*X[:,1:2]
    w1 = w[0,:].reshape((1,10))
    w2 = w[1,:].reshape((1,10))
    b = b.reshape((1,10))
    sig = lambda x,y: 1./(1.+np.exp(-(x*w1)-(y*w2)-b))
    mydSig = egrad(egrad(egrad(egrad(egrad(sig,0),0),1),1),1)

    Fp1 = sig(x,y)
    Fp2 = onp.delete(mydSig(x,y),nC2[0],axis=1)

    assert(np.linalg.norm(Fc1-Fp1,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc2-Fp2,ord='fro') < 1e-13)

def test_nELMTanh():
    dim = 2
    nC = -1*np.ones(1,dtype=np.int32)
    d = np.zeros(dim,dtype=np.int32)
    c = np.ones(dim)
    d2 = np.array([2,3],dtype=np.int32)
    nC2 = np.array([4],dtype=np.int32)
    n = np.array([10]*dim)
    N = np.prod(n)
    z = np.linspace(0,5,num=n[0])
    X = onp.zeros((N,dim))
    for k in range(dim):
        nProd = np.prod(n[k+1:])
        nStack = np.prod(n[0:k])
        dark = np.hstack([z]*nProd)
        X[:,k] = onp.array([dark]*nStack).flatten()

    useVal = np.array([0,0],dtype=np.int32)
    elm1 = nELMTanh(X.T,nC,10,c)
    w = elm1.w
    b = elm1.b
    elm2 = nELMTanh(X.T,nC2,10,c)
    elm2.w = w
    elm2.b = b
    Fc1 = elm1.H(X.T,d,False,useVal)
    Fc2 = elm2.H(X.T,d2,False,useVal)

    x = np.ones((100,10))*X[:,0:1]
    y = np.ones((100,10))*X[:,1:2]
    w1 = w[0,:].reshape((1,10))
    w2 = w[1,:].reshape((1,10))
    b = b.reshape((1,10))
    tanh = lambda x,y: np.tanh(w1*x + w2*y + b)
    mydTanh = egrad(egrad(egrad(egrad(egrad(tanh,0),0),1),1),1)

    Fp1 = tanh(x,y)
    Fp2 = onp.delete(mydTanh(x,y),nC2[0],axis=1)

    assert(np.linalg.norm(Fc1-Fp1,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc2-Fp2,ord='fro') < 1e-13)

def test_nELMSin():
    dim = 2
    nC = -1*np.ones(1,dtype=np.int32)
    d = np.zeros(dim,dtype=np.int32)
    c = np.ones(dim)
    d2 = np.array([2,3],dtype=np.int32)
    nC2 = np.array([4],dtype=np.int32)
    n = np.array([10]*dim)
    N = np.prod(n)
    z = np.linspace(0,5,num=n[0])
    X = onp.zeros((N,dim))
    for k in range(dim):
        nProd = np.prod(n[k+1:])
        nStack = np.prod(n[0:k])
        dark = np.hstack([z]*nProd)
        X[:,k] = onp.array([dark]*nStack).flatten()

    useVal = np.array([0,0],dtype=np.int32)
    elm1 = nELMSin(X.T,nC,10,c)
    w = elm1.w
    b = elm1.b
    elm2 = nELMSin(X.T,nC2,10,c)
    elm2.w = w
    elm2.b = b
    Fc1 = elm1.H(X.T,d,False,useVal)
    Fc2 = elm2.H(X.T,d2,False,useVal)

    x = np.ones((100,10))*X[:,0:1]
    y = np.ones((100,10))*X[:,1:2]
    w1 = w[0,:].reshape((1,10))
    w2 = w[1,:].reshape((1,10))
    b = b.reshape((1,10))
    Sin = lambda x,y: np.sin(w1*x + w2*y + b)
    mydSin = egrad(egrad(egrad(egrad(egrad(Sin,0),0),1),1),1)

    Fp1 = Sin(x,y)
    Fp2 = onp.delete(mydSin(x,y),nC2[0],axis=1)

    assert(np.linalg.norm(Fc1-Fp1,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc2-Fp2,ord='fro') < 1e-12)

def test_nELMSwish():
    dim = 2
    nC = -1*np.ones(1,dtype=np.int32)
    d = np.zeros(dim,dtype=np.int32)
    c = np.ones(dim)
    d2 = np.array([2,3],dtype=np.int32)
    nC2 = np.array([4],dtype=np.int32)
    n = np.array([10]*dim)
    N = np.prod(n)
    z = np.linspace(0,5,num=n[0])
    X = onp.zeros((N,dim))
    for k in range(dim):
        nProd = np.prod(n[k+1:])
        nStack = np.prod(n[0:k])
        dark = np.hstack([z]*nProd)
        X[:,k] = onp.array([dark]*nStack).flatten()

    useVal = np.array([0,0],dtype=np.int32)
    elm1 = nELMSwish(X.T,nC,10,c)
    w = elm1.w
    b = elm1.b
    elm2 = nELMSwish(X.T,nC2,10,c)
    elm2.w = w
    elm2.b = b
    Fc1 = elm1.H(X.T,d,False,useVal)
    Fc2 = elm2.H(X.T,d2,False,useVal)

    x = np.ones((100,10))*X[:,0:1]
    y = np.ones((100,10))*X[:,1:2]
    w1 = w[0,:].reshape((1,10))
    w2 = w[1,:].reshape((1,10))
    b = b.reshape((1,10))
    swish = lambda x,y:  (w1*x + w2*y  + b) * (1./(1.+np.exp(-(x*w1)-(y*w2)-b)))
    mydswish = egrad(egrad(egrad(egrad(egrad(swish,0),0),1),1),1)

    Fp1 = swish(x,y)
    Fp2 = onp.delete(mydswish(x,y),nC2[0],axis=1)

    assert(np.linalg.norm(Fc1-Fp1,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc2-Fp2,ord='fro') < 1e-12)
