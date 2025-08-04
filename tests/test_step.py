import jax.numpy as np

from tfc.utils import egrad, step

def test_step():
    A = np.array([5.,4.,-1.,1.,0.])
    B = step(A)
    C = step(A)
    assert(np.all(B == np.array([1.,1.,0.,1.,0.])))
    assert(np.all(B == C))

    A = np.array([[5.,4.,-1.,1.,0.],[-1.5,0.,0.,2.2,5.6]])
    B = step(A)
    C = step(A)
    assert(np.all(B == np.array([[1.,1.,0.,1.,0.],[0.,0.,0.,1.,1.]])))
    assert(np.all(B == C))

def test_dstep():
    A = np.array([5.,4.,-1.,1.,0.])
    B = lambda A: step(A)
    C = lambda A: step(A)
    D = egrad(B,0)(A)
    E = egrad(C,0)(A)
    assert(np.all(D == np.zeros_like(D)))
    assert(np.all(E == D))

    A = np.array([[5.,4.,-1.,1.,0.],[-1.5,0.,0.,2.2,5.6]])
    D = egrad(B,0)(A)
    E = egrad(C,0)(A)
    assert(np.all(D == np.zeros_like(D)))
    assert(np.all(E == D))

    B = lambda A: A*step(A)+A
    C = lambda A: A*step(A)+A
    D = egrad(B,0)(A)
    E = egrad(C,0)(A)
    true = np.array([[1.,1.,0.,1.,0.],[0.,0.,0.,1.,1.]])+np.ones_like(A)
    assert(np.all(D == true))
    assert(np.all(E == D))

    B = lambda A: A**3*step(A)+np.sin(A)*step(A)
    C = lambda A: A**3*step(A)+np.sin(A)*step(A)
    D = egrad(egrad(B,0),0)(A)
    E = egrad(egrad(C,0),0)(A)
    true = 6.*A*np.array([[1.,1.,0.,1.,0.],[0.,0.,0.,1.,1.]])-np.sin(A)*np.array([[1.,1.,0.,1.,0.],[0.,0.,0.,1.,1.]])
    assert(np.all(D == true))
    assert(np.all(E == D))

