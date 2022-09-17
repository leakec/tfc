from numpy import random
random.seed(0)

import jax.numpy as np
from tfc import utfc
from tfc.utils import egrad, LS

def test_complex_ODE():
    # Domain circle of radius 1 around 1+1.j
    # Problem: f'(x) = f(x) s.t. f(1+1.j) = 5.0 + 0.j
    # Real solution: 5.0 * exp(z) / exp(1+1.j)

    # Store the real solution
    realSoln = lambda x: 5.0 * np.exp(x) / np.exp(1+1.j)

    # Create UTFC class
    a = np.sin(np.pi/4.)
    x0 = 1-a + (1-a)*1.j
    xf = 1+a + (1+a)*1.j
    tfc = utfc(100, 0, 90, basis="ELMTanh", x0=x0, xf=xf, backend="Python")
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
    u = lambda x,xi: g(x,xi) + 5.0 - g((1+1.j)*np.ones_like(x),xi)

    # Create loss function
    ud = egrad(u)
    L = lambda xi,x: ud(x,xi) - u(x,xi)

    # Solve the problem
    xi = np.zeros(H(x).shape[1])
    xi = LS(xi,L,x,constant_arg_nums=[1], method="lstsq")

    # Create test points
    numTest = 30
    r = np.linspace(0.,1.,numTest).reshape((1,numTest))
    th = np.linspace(0.,2*np.pi,numTest).reshape((numTest,1))
    x = r*np.sin(th)
    y = r*np.cos(th)*1.j
    test = (x+y).flatten() + 1. + 1.j

    # Calculate the error
    U = u(test,xi)
    err = U - realSoln(test)
    maxErr = np.max(np.abs(err))

    # Check results
    assert(maxErr < 1e-8)
