import jax.numpy as np

from tfc.utils.BF import (
    CP,
    LeP,
    LaP,
    HoPpro,
    HoPphy,
    FS,
    ELMReLU,
    ELMSigmoid,
    ELMTanh,
    ELMSin,
    ELMSwish,
)
from tfc.utils import egrad


def test_CP():
    from tfc.utils.BF.BF_Py import CP as pCP

    x = np.linspace(0, 2, num=10)

    cp1 = CP(0.0, 2.0, np.array([], dtype=np.int32), 5)
    cp2 = CP(0.0, 2.0, np.array([], dtype=np.int32), 10)
    Fc1 = cp1.H(x, 0, False)
    Fc2 = cp2.H(x, 3, False)

    pcp1 = pCP(0.0, 2.0, np.array([], dtype=np.int32), 5)
    pcp2 = pCP(0.0, 2.0, np.array([], dtype=np.int32), 10)
    Fp1 = pcp1.H(x, d=0, full=False)
    Fp2 = pcp2.H(x, d=3, full=False)

    assert np.linalg.norm(Fc1 - Fp1, ord="fro") < 1e-14
    assert np.linalg.norm(Fc2 - Fp2, ord="fro") < 1e-14


def test_LeP():
    from tfc.utils.BF.BF_Py import LeP as pLeP

    x = np.linspace(0, 2, num=10)
    lep1 = LeP(0.0, 2.0, np.array([], dtype=np.int32), 5)
    lep2 = LeP(0.0, 2.0, np.array([], dtype=np.int32), 10)
    Fc1 = lep1.H(x, 0, False)
    Fc2 = lep2.H(x, 3, False)

    plep1 = pLeP(0.0, 2.0, np.array([], dtype=np.int32), 5)
    plep2 = pLeP(0.0, 2.0, np.array([], dtype=np.int32), 10)
    Fp1 = plep1.H(x, d=0, full=False)
    Fp2 = plep2.H(x, d=3, full=False)

    assert np.linalg.norm(Fc1 - Fp1, ord="fro") < 1e-14
    assert np.linalg.norm(Fc2 - Fp2, ord="fro") < 1e-14


def test_LaP():
    from tfc.utils.BF.BF_Py import LaP as pLaP

    x = np.linspace(0, 5, num=10)
    lap1 = LaP(0.0, 5.0, np.array([], dtype=np.int32), 5)
    lap2 = LaP(0.0, 5.0, np.array([], dtype=np.int32), 10)
    Fc1 = lap1.H(x, 0, False)
    Fc2 = lap2.H(x, 3, False)

    plap1 = pLaP(0.0, 5.0, np.array([], dtype=np.int32), 5)
    plap2 = pLaP(0.0, 5.0, np.array([], dtype=np.int32), 10)
    Fp1 = plap1.H(x, d=0, full=False)
    Fp2 = plap2.H(x, d=3, full=False)

    assert np.linalg.norm(Fc1 - Fp1, ord="fro") < 1e-14
    assert np.linalg.norm(Fc2 - Fp2, ord="fro") < 1e-14


def test_HoPpro():
    from tfc.utils.BF.BF_Py import HoPpro as pHoPpro

    x = np.linspace(0, 5, num=10)
    hoppro1 = HoPpro(0.0, 5.0, np.array([], dtype=np.int32), 5)
    hoppro2 = HoPpro(0.0, 5.0, np.array([], dtype=np.int32), 10)
    Fc1 = hoppro1.H(x, 0, False)
    Fc2 = hoppro2.H(x, 3, False)

    phoppro1 = pHoPpro(0.0, 5.0, np.array([], dtype=np.int32), 5)
    phoppro2 = pHoPpro(0.0, 5.0, np.array([], dtype=np.int32), 10)
    Fp1 = phoppro1.H(x, 0, full=False)
    Fp2 = phoppro2.H(x, 3, full=False)

    assert np.linalg.norm(Fc1 - Fp1, ord="fro") < 1e-14
    assert np.linalg.norm(Fc2 - Fp2, ord="fro") < 1e-14


def test_HoPphy():
    from tfc.utils.BF.BF_Py import HoPphy as pHoPphy

    x = np.linspace(0, 5, num=10)
    hopphy1 = HoPphy(0.0, 5.0, np.array([], dtype=np.int32), 5)
    hopphy2 = HoPphy(0.0, 5.0, np.array([], dtype=np.int32), 10)
    Fc1 = hopphy1.H(x, 0, False)
    Fc2 = hopphy2.H(x, 3, False)

    phopphy1 = pHoPphy(0.0, 5.0, np.array([], dtype=np.int32), 5)
    phopphy2 = pHoPphy(0.0, 5.0, np.array([], dtype=np.int32), 10)
    Fp1 = phopphy1.H(x, d=0, full=False)
    Fp2 = phopphy2.H(x, d=3, full=False)

    assert np.linalg.norm(Fc1 - Fp1, ord="fro") < 1e-14
    assert np.linalg.norm(Fc2 - Fp2, ord="fro") < 1e-14


def test_FS():
    from tfc.utils.BF.BF_Py import FS as pFS

    x = np.linspace(0, 2 * np.pi, num=10)
    fs1 = FS(0.0, 2.0 * np.pi, np.array([], dtype=np.int32), 5)
    fs2 = FS(0.0, 2.0 * np.pi, np.array([], dtype=np.int32), 10)
    Fc1 = fs1.H(x, 0, False)
    Fc2 = fs2.H(x, 1, False)
    Fc3 = fs2.H(x, 2, False)
    Fc4 = fs2.H(x, 3, False)
    Fc5 = fs2.H(x, 4, False)

    pfs1 = pFS(0.0, 2.0 * np.pi, np.array([], dtype=np.int32), 5)
    pfs2 = pFS(0.0, 2.0 * np.pi, np.array([], dtype=np.int32), 10)
    Fp1 = pfs1.H(x, d=0, full=False)
    Fp2 = pfs2.H(x, d=1, full=False)
    Fp3 = pfs2.H(x, d=2, full=False)
    Fp4 = pfs2.H(x, d=3, full=False)
    Fp5 = pfs2.H(x, d=4, full=False)

    assert np.linalg.norm(Fc1 - Fp1, ord="fro") < 1e-14
    assert np.linalg.norm(Fc2 - Fp2, ord="fro") < 1e-14
    assert np.linalg.norm(Fc3 - Fp3, ord="fro") < 1e-14
    assert np.linalg.norm(Fc4 - Fp4, ord="fro") < 5e-14
    assert np.linalg.norm(Fc5 - Fp5, ord="fro") < 5e-13


def test_ELMReLU():
    from tfc.utils.BF.BF_Py import ELMReLU as pELMReLU

    x = np.linspace(0, 1, num=10)
    elm = ELMReLU(0.0, 1.0, np.array([], dtype=np.int32), 10)
    Fc1 = elm.H(x, 0, False)
    Fc2 = elm.H(x, 1, False)
    Fc3 = elm.H(x, 2, False)

    pelm = pELMReLU(0.0, 1.0, np.array([], dtype=np.int32), 10)
    pelm.w = elm.w
    pelm.b = elm.b

    Fp1 = pelm.H(x, d=0, full=False)
    Fp2 = pelm.H(x, d=1, full=False)
    Fp3 = pelm.H(x, d=2, full=False)

    assert np.linalg.norm(Fc1 - Fp1, ord="fro") < 1e-14
    assert np.linalg.norm(Fc2 - Fp2, ord="fro") < 1e-14
    assert np.linalg.norm(Fc3 - Fp3, ord="fro") < 1e-14


def test_ELMSigmoid():
    from tfc.utils.BF.BF_Py import ELMSigmoid as pELMSigmoid

    x = np.linspace(0, 1, num=10)
    elm = ELMSigmoid(0.0, 1.0, np.array([], dtype=np.int32), 10)
    Fc1 = elm.H(x, 0, False)
    Fc2 = elm.H(x, 1, False)
    Fc3 = elm.H(x, 2, False)
    Fc4 = elm.H(x, 3, False)

    pelm = pELMSigmoid(0.0, 1.0, np.array([], dtype=np.int32), 10)
    pelm.w = elm.w
    pelm.b = elm.b

    Fp1 = pelm.H(x, 0, False)
    Fp2 = pelm.H(x, 1, False)
    Fp3 = pelm.H(x, 2, False)
    Fp4 = pelm.H(x, 3, False)

    assert np.linalg.norm(Fc1 - Fp1, ord="fro") < 1e-14
    assert np.linalg.norm(Fc2 - Fp2, ord="fro") < 1e-14
    assert np.linalg.norm(Fc3 - Fp3, ord="fro") < 5e-10
    assert np.linalg.norm(Fc4 - Fp4, ord="fro") < 5e-10


def test_ELMTanh():
    from tfc.utils.BF.BF_Py import ELMTanh as pELMTanh

    x = np.linspace(0, 1, num=10)
    elm = ELMTanh(0.0, 1.0, np.array([], dtype=np.int32), 10)
    Fc1 = elm.H(x, 0, False)
    Fc2 = elm.H(x, 1, False)
    Fc3 = elm.H(x, 2, False)
    Fc4 = elm.H(x, 3, False)

    pelm = pELMTanh(0.0, 1.0, np.array([], dtype=np.int32), 10)
    pelm.w = elm.w
    pelm.b = elm.b
    Fp1 = pelm.H(x, 0, False)
    Fp2 = pelm.H(x, 1, False)
    Fp3 = pelm.H(x, 2, False)
    Fp4 = pelm.H(x, 3, False)

    assert np.linalg.norm(Fc1 - Fp1, ord="fro") < 1e-14
    assert np.linalg.norm(Fc2 - Fp2, ord="fro") < 5e-14
    assert np.linalg.norm(Fc3 - Fp3, ord="fro") < 5e-13
    assert np.linalg.norm(Fc4 - Fp4, ord="fro") < 5e-10


def test_ELMSin():
    from tfc.utils.BF.BF_Py import ELMSin as pELMSin

    x = np.linspace(0, 1, num=10)
    elm = ELMSin(0.0, 1.0, np.array([], dtype=np.int32), 10)
    Fc1 = elm.H(x, 0, False)
    Fc2 = elm.H(x, 1, False)
    Fc3 = elm.H(x, 2, False)
    Fc4 = elm.H(x, 3, False)
    Fc5 = elm.H(x, 4, False)

    pelm = pELMSin(0.0, 1.0, np.array([], dtype=np.int32), 10)
    pelm.w = elm.w
    pelm.b = elm.b
    Fp1 = elm.H(x, 0, False)
    Fp2 = elm.H(x, 1, False)
    Fp3 = elm.H(x, 2, False)
    Fp4 = elm.H(x, 3, False)
    Fp5 = elm.H(x, 4, False)

    assert np.linalg.norm(Fc1 - Fp1, ord="fro") < 1e-14
    assert np.linalg.norm(Fc2 - Fp2, ord="fro") < 1e-14
    assert np.linalg.norm(Fc3 - Fp3, ord="fro") < 5e-14
    assert np.linalg.norm(Fc4 - Fp4, ord="fro") < 5e-12
    assert np.linalg.norm(Fc5 - Fp5, ord="fro") < 5e-12


def test_ELMSwish():
    from tfc.utils.BF.BF_Py import ELMSwish as pELMSwish

    x = np.linspace(0, 1, num=10)
    elm = ELMSwish(0.0, 1.0, np.array([], dtype=np.int32), 10)
    Fc1 = elm.H(x, 0, False)
    Fc2 = elm.H(x, 1, False)
    Fc3 = elm.H(x, 2, False)
    Fc4 = elm.H(x, 3, False)
    Fc5 = elm.H(x, 4, False)

    pelm = pELMSwish(0.0, 1.0, np.array([], dtype=np.int32), 10)
    pelm.w = elm.w
    pelm.b = elm.b
    Fp1 = pelm.H(x, 0, False)
    Fp2 = pelm.H(x, 1, False)
    Fp3 = pelm.H(x, 2, False)
    Fp4 = pelm.H(x, 3, False)
    Fp5 = pelm.H(x, 4, False)

    assert np.linalg.norm(Fc1 - Fp1, ord="fro") < 1e-14
    assert np.linalg.norm(Fc2 - Fp2, ord="fro") < 1e-13
    assert np.linalg.norm(Fc3 - Fp3, ord="fro") < 5e-10
    assert np.linalg.norm(Fc4 - Fp4, ord="fro") < 5e-10
    assert np.linalg.norm(Fc5 - Fp5, ord="fro") < 5e-9
