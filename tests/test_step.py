import jax.numpy as np

from tfc import tfc as TFC
from tfc import ntfc as nTFC

def test_step():
    A = np.array([5.,4.,-1.,1.,0.])
    tfc = TFC(10,0,3,basis='CP',x0=-1.,xf=1.)
    ntfc = nTFC(np.array([5,5]),np.array([0,0]),3,dim=2,basis='CP',x0=[-1.,-1.],xf=[1.,1.])
    B = tfc.step(A)
    C = ntfc.step(A)
    assert(np.all(B == np.array([1.,1.,0.,1.,0.])))
    assert(np.all(B == C))

    A = np.array([[5.,4.,-1.,1.,0.],[-1.5,0.,0.,2.2,5.6]])
    B = tfc.step(A)
    C = ntfc.step(A)
    assert(np.all(B == np.array([[1.,1.,0.,1.,0.],[0.,0.,0.,1.,1.]])))
    assert(np.all(B == C))

def test_dstep():
    A = np.array([5.,4.,-1.,1.,0.])
    tfc = TFC(10,0,3,basis='CP',x0=-1.,xf=1.)
    ntfc = nTFC(np.array([5,5]),np.array([0,0]),3,dim=2,basis='CP',x0=[-1.,-1.],xf=[1.,1.])
    B = lambda A: tfc.step(A)
    C = lambda A: ntfc.step(A)
    D = tfc.egrad(B,0)(A)
    E = tfc.egrad(C,0)(A)
    assert(np.all(D == np.zeros_like(D)))
    assert(np.all(E == D))

    A = np.array([[5.,4.,-1.,1.,0.],[-1.5,0.,0.,2.2,5.6]])
    D = tfc.egrad(B,0)(A)
    E = tfc.egrad(C,0)(A)
    assert(np.all(D == np.zeros_like(D)))
    assert(np.all(E == D))

    B = lambda A: A*tfc.step(A)+A
    C = lambda A: A*ntfc.step(A)+A
    D = tfc.egrad(B,0)(A)
    E = tfc.egrad(C,0)(A)
    true = np.array([[1.,1.,0.,1.,0.],[0.,0.,0.,1.,1.]])+np.ones_like(A)
    assert(np.all(D == true))
    assert(np.all(E == D))

    B = lambda A: A**3*tfc.step(A)+np.sin(A)*tfc.step(A)
    C = lambda A: A**3*ntfc.step(A)+np.sin(A)*ntfc.step(A)
    D = tfc.egrad(tfc.egrad(B,0),0)(A)
    E = tfc.egrad(tfc.egrad(C,0),0)(A)
    true = 6.*A*np.array([[1.,1.,0.,1.,0.],[0.,0.,0.,1.,1.]])-np.sin(A)*np.array([[1.,1.,0.,1.,0.],[0.,0.,0.,1.,1.]])
    assert(np.all(D == true))
    assert(np.all(E == D))

