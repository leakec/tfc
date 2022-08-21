from jax.config import config
config.update('jax_enable_x64', True)
import jax.numpy as np

from tfc.utils.BF import CP, LeP, LaP, HoPpro, HoPphy, FS, ELMReLU, ELMSigmoid, ELMTanh, ELMSin, ELMSwish
from tfc.utils import egrad

def test_CP():
    from tfc.utils.BF.BF_Py import CP as pCP
    x = np.linspace(0,2,num=10)

    cp1 = CP(0.,2.,np.array([],dtype=np.int32),5)
    cp2 = CP(0.,2.,np.array([],dtype=np.int32),10)
    Fc1 = cp1.H(x,0,False)
    Fc2 = cp2.H(x,3,False)

    pcp1 = pCP(0.,2.,np.array([],dtype=np.int32),5)
    pcp2 = pCP(0.,2.,np.array([],dtype=np.int32),10)
    Fp1 = pcp1.H(x,d=0,full=False)
    Fp2 = pcp2.H(x,d=3,full=False)

    assert(np.linalg.norm(Fc1-Fp1,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc2-Fp2,ord='fro') < 1e-14)

def test_LeP():
    from tfc.utils.BF.BF_Py import LeP as pLeP
    x = np.linspace(0,2,num=10)
    lep1 = LeP(0.,2.,np.array([],dtype=np.int32),5)
    lep2 = LeP(0.,2.,np.array([],dtype=np.int32),10)
    Fc1 = lep1.H(x,0,False)
    Fc2 = lep2.H(x,3,False)

    plep1 = pLeP(0.,2.,np.array([],dtype=np.int32),5)
    plep2 = pLeP(0.,2.,np.array([],dtype=np.int32),10)
    Fp1 = plep1.H(x,d=0,full=False)
    Fp2 = plep2.H(x,d=3,full=False)

    assert(np.linalg.norm(Fc1-Fp1,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc2-Fp2,ord='fro') < 1e-14)

def test_LaP():
    from tfc.utils.BF.BF_Py import LaP as pLaP
    x = np.linspace(0,5,num=10)
    lap1 = LaP(0.,5.,np.array([],dtype=np.int32),5)
    lap2 = LaP(0.,5.,np.array([],dtype=np.int32),10)
    Fc1 = lap1.H(x,0,False)
    Fc2 = lap2.H(x,3,False)

    plap1 = pLaP(0.,5.,np.array([],dtype=np.int32),5)
    plap2 = pLaP(0.,5.,np.array([],dtype=np.int32),10)
    Fp1 = plap1.H(x,d=0,full=False)
    Fp2 = plap2.H(x,d=3,full=False)

    assert(np.linalg.norm(Fc1-Fp1,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc2-Fp2,ord='fro') < 1e-14)

def test_HoPpro():
    from tfc.utils.BF.BF_Py import HoPpro as pHoPpro
    x = np.linspace(0,5,num=10)
    hoppro1 = HoPpro(0.,5.,np.array([],dtype=np.int32),5)
    hoppro2 = HoPpro(0.,5.,np.array([],dtype=np.int32),10)
    Fc1 = hoppro1.H(x,0,False)
    Fc2 = hoppro2.H(x,3,False)

    phoppro1 = pHoPpro(0.,5.,np.array([],dtype=np.int32),5)
    phoppro2 = pHoPpro(0.,5.,np.array([],dtype=np.int32),10)
    Fp1 = phoppro1.H(x,0,full=False)
    Fp2 = phoppro2.H(x,3,full=False)

    assert(np.linalg.norm(Fc1-Fp1,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc2-Fp2,ord='fro') < 1e-14)

def test_HoPphy():
    from tfc.utils.BF.BF_Py import HoPphy as pHoPphy
    x = np.linspace(0,5,num=10)
    hopphy1 = HoPphy(0.,5.,np.array([],dtype=np.int32),5)
    hopphy2 = HoPphy(0.,5.,np.array([],dtype=np.int32),10)
    Fc1 = hopphy1.H(x,0,False)
    Fc2 = hopphy2.H(x,3,False)

    phopphy1 = pHoPphy(0.,5.,np.array([],dtype=np.int32),5)
    phopphy2 = pHoPphy(0.,5.,np.array([],dtype=np.int32),10)
    Fp1 = phopphy1.H(x,d=0,full=False)
    Fp2 = phopphy2.H(x,d=3,full=False)

    assert(np.linalg.norm(Fc1-Fp1,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc2-Fp2,ord='fro') < 1e-14)

def test_FS():
    from tfc.utils.BF.BF_Py import FS as pFS
    x = np.linspace(0,2*np.pi,num=10)
    fs1 = FS(0.,2.*np.pi,np.array([],dtype=np.int32),5)
    fs2 = FS(0.,2.*np.pi,np.array([],dtype=np.int32),10)
    Fc1 = fs1.H(x,0,False)
    Fc2 = fs2.H(x,1,False)
    Fc3 = fs2.H(x,2,False)
    Fc4 = fs2.H(x,3,False)
    Fc5 = fs2.H(x,4,False)

    pfs1 = pFS(0.,2.*np.pi,np.array([],dtype=np.int32),5)
    pfs2 = pFS(0.,2.*np.pi,np.array([],dtype=np.int32),10)
    Fp1 = pfs1.H(x,d=0,full=False)
    Fp2 = pfs2.H(x,d=1,full=False)
    Fp3 = pfs2.H(x,d=2,full=False)
    Fp4 = pfs2.H(x,d=3,full=False)
    Fp5 = pfs2.H(x,d=4,full=False)

    assert(np.linalg.norm(Fc1-Fp1,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc2-Fp2,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc3-Fp3,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc4-Fp4,ord='fro') < 5e-14)
    assert(np.linalg.norm(Fc5-Fp5,ord='fro') < 5e-13)

def test_ELMReLU():
    from jax.nn import relu as ReLU
    x = np.linspace(0,1,num=10)
    elm = ELMReLU(0.,1.,np.array([],dtype=np.int32),10)
    Fc1 = elm.H(x,0,False)
    Fc2 = elm.H(x,1,False)
    Fc3 = elm.H(x,2,False)
    Fc4 = elm.H(x,3,False)

    x = x.reshape(10,1)
    x = np.ones((10,10))*x
    w = elm.w.reshape(1,10)
    b = elm.b.reshape(1,10)
    relu = lambda x: ReLU(w*x+b)
    drelu = egrad(relu)
    d2relu = egrad(drelu)

    Fp1 = relu(x)
    Fp2 = drelu(x)
    Fp3 = d2relu(x)

    assert(np.linalg.norm(Fc1-Fp1,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc2-Fp2,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc3-Fp3,ord='fro') < 1e-14)

def test_ELMSigmoid():
    x = np.linspace(0,1,num=10)
    elm = ELMSigmoid(0.,1.,np.array([],dtype=np.int32),10)
    Fc1 = elm.H(x,0,False)
    Fc2 = elm.H(x,1,False)
    Fc3 = elm.H(x,2,False)
    Fc4 = elm.H(x,3,False)
    Fc5 = elm.H(x,4,False)
    Fc6 = elm.H(x,5,False)
    Fc7 = elm.H(x,6,False)
    Fc8 = elm.H(x,7,False)
    Fc9 = elm.H(x,8,False)

    x = x.reshape(10,1)
    x = np.ones((10,10))*x
    w = elm.w.reshape(1,10)
    b = elm.b.reshape(1,10)
    sig = lambda x: 1./(1.+np.exp(-w*x-b))
    dsig = egrad(sig)
    d2sig = egrad(dsig)
    d3sig = egrad(d2sig)
    d4sig = egrad(d2sig)
    #d5sig = egrad(d2sig)
    #d6sig = egrad(d2sig)
    #d7sig = egrad(d2sig)
    #d8sig = egrad(d2sig)
    #d9sig = egrad(d2sig)

    Fp1 = sig(x)
    Fp2 = dsig(x)
    Fp3 = d2sig(x)
    Fp4 = d3sig(x)
    #Fp5 = d4sig(x)
    #Fp6 = d5sig(x)
    #Fp7 = d6sig(x)
    #Fp8 = d7sig(x)
    #Fp9 = d8sig(x)

    assert(np.linalg.norm(Fc1-Fp1,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc2-Fp2,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc3-Fp3,ord='fro') < 5e-10)
    assert(np.linalg.norm(Fc4-Fp4,ord='fro') < 5e-10)
    #assert(np.linalg.norm(Fc5-Fp5,ord='fro') < 5e-10)
    #assert(np.linalg.norm(Fc6-Fp6,ord='fro') < 1e-10)
    #assert(np.linalg.norm(Fc7-Fp7,ord='fro') < 1e-10)
    #assert(np.linalg.norm(Fc8-Fp8,ord='fro') < 5e-10)
    #assert(np.linalg.norm(Fc9-Fp9,ord='fro') < 1e-12)

def test_ELMTanh():
    x = np.linspace(0,1,num=10)
    elm = ELMTanh(0.,1.,np.array([],dtype=np.int32),10)
    Fc1 = elm.H(x,0,False)
    Fc2 = elm.H(x,1,False)
    Fc3 = elm.H(x,2,False)
    Fc4 = elm.H(x,3,False)
    Fc5 = elm.H(x,4,False)
    Fc6 = elm.H(x,5,False)
    Fc7 = elm.H(x,6,False)
    Fc8 = elm.H(x,7,False)
    Fc9 = elm.H(x,8,False)

    x = x.reshape(10,1)
    x = np.ones((10,10))*x
    w = elm.w.reshape(1,10)
    b = elm.b.reshape(1,10)
    Tanh = lambda x: np.tanh(w*x + b)
    dTanh = egrad(Tanh)
    d2Tanh = egrad(dTanh)
    d3Tanh = egrad(d2Tanh)
    d4Tanh = egrad(d3Tanh)
    #d5Tanh = egrad(d4Tanh)
    #d6Tanh = egrad(d5Tanh)
    #d7Tanh = egrad(d6Tanh)
    #d8Tanh = egrad(d7Tanh)
    #d9Tanh = egrad(d8Tanh)

    Fp1 = Tanh(x)
    Fp2 = dTanh(x)
    Fp3 = d2Tanh(x)
    Fp4 = d3Tanh(x)
    #Fp5 = d4Tanh(x)
    #Fp6 = d5Tanh(x)
    #Fp7 = d6Tanh(x)
    #Fp8 = d7Tanh(x)
    #Fp9 = d8Tanh(x)

    assert(np.linalg.norm(Fc1-Fp1,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc2-Fp2,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc3-Fp3,ord='fro') < 1e-13)
    assert(np.linalg.norm(Fc4-Fp4,ord='fro') < 5e-10)
    #assert(np.linalg.norm(Fc5-Fp5,ord='fro') < 5e-10)
    #assert(np.linalg.norm(Fc6-Fp6,ord='fro') < 1e-9)
    #assert(np.linalg.norm(Fc7-Fp7,ord='fro') < 5e-9)
    #assert(np.linalg.norm(Fc8-Fp8,ord='fro') < 5e-9)
    #assert(np.linalg.norm(Fc9-Fp9,ord='fro') < 1e-12)

def test_ELMSin():
    x = np.linspace(0,1,num=10)
    elm = ELMSin(0.,1.,np.array([],dtype=np.int32),10)
    Fc1 = elm.H(x,0,False)
    Fc2 = elm.H(x,1,False)
    Fc3 = elm.H(x,2,False)
    Fc4 = elm.H(x,3,False)
    Fc5 = elm.H(x,4,False)

    x = x.reshape(10,1)
    x = np.ones((10,10))*x
    w = elm.w.reshape(1,10)
    b = elm.b.reshape(1,10)
    sin = lambda x: np.sin(w*x + b)
    dsin = egrad(sin)
    d2sin = egrad(dsin)
    d3sin = egrad(d2sin)
    d4sin = egrad(d3sin)
    d5sin = egrad(d4sin)

    Fp1 = sin(x)
    Fp2 = dsin(x)
    Fp3 = d2sin(x)
    Fp4 = d3sin(x)
    Fp5 = d4sin(x)

    assert(np.linalg.norm(Fc1-Fp1,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc2-Fp2,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc3-Fp3,ord='fro') < 5e-14)
    assert(np.linalg.norm(Fc4-Fp4,ord='fro') < 5e-12)
    assert(np.linalg.norm(Fc5-Fp5,ord='fro') < 5e-12)

def test_ELMSwish():
    x = np.linspace(0,1,num=10)
    elm = ELMSwish(0.,1.,np.array([],dtype=np.int32),10)
    Fc1 = elm.H(x,0,False)
    Fc2 = elm.H(x,1,False)
    Fc3 = elm.H(x,2,False)
    Fc4 = elm.H(x,3,False)
    Fc5 = elm.H(x,4,False)
    Fc6 = elm.H(x,5,False)
    Fc7 = elm.H(x,6,False)
    Fc8 = elm.H(x,7,False)
    Fc9 = elm.H(x,8,False)

    x = x.reshape(10,1)
    x = np.ones((10,10))*x
    w = elm.w.reshape(1,10)
    b = elm.b.reshape(1,10)
    swish = lambda x: (w*x+b) * (1./(1.+np.exp(-w*x-b)))
    dswish = egrad(swish)
    d2swish = egrad(dswish)
    d3swish = egrad(d2swish)
    d4swish = egrad(d3swish)
    d5swish = egrad(d4swish)
    #d6swish = egrad(d5swish)
    #d7swish = egrad(d6swish)
    #d8swish = egrad(d7swish)
    #d9swish = egrad(d8swish)

    Fp1 = swish(x)
    Fp2 = dswish(x)
    Fp3 = d2swish(x)
    Fp4 = d3swish(x)
    Fp5 = d4swish(x)
    #Fp6 = d5swish(x)
    #Fp7 = d6swish(x)
    #Fp8 = d7swish(x)
    #Fp9 = d8swish(x)

    assert(np.linalg.norm(Fc1-Fp1,ord='fro') < 1e-14)
    assert(np.linalg.norm(Fc2-Fp2,ord='fro') < 1e-13)
    assert(np.linalg.norm(Fc3-Fp3,ord='fro') < 5e-10)
    assert(np.linalg.norm(Fc4-Fp4,ord='fro') < 5e-10)
    assert(np.linalg.norm(Fc5-Fp5,ord='fro') < 5e-9)
    #assert(np.linalg.norm(Fc6-Fp6,ord='fro') < 5e-9)
    #assert(np.linalg.norm(Fc7-Fp7,ord='fro') < 5e-9)
    #assert(np.linalg.norm(Fc8-Fp8,ord='fro') < 1e-9)
    # assert(np.linalg.norm(Fc9-Fp9,ord='fro') < 1e-12)
