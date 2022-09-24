from jax.config import config

config.update("jax_enable_x64", True)
import numpy as onp
import jax.numpy as np

from tfc.utils.BF import nCP, nLeP, nFS, nELMSigmoid, nELMTanh, nELMSin, nELMSwish, nELMReLU


def test_nCP():
    from tfc.utils.BF.BF_Py import nCP as pnCP

    dim = 2
    nC = -1 * np.ones((dim, 1), dtype=np.int32)
    d = np.zeros(dim, dtype=np.int32)
    c = np.ones(dim)
    d2 = np.array([2, 3], dtype=np.int32)
    nC2 = np.block([[np.arange(4), -1.0 * np.ones(3)], [np.arange(7)]]).astype(np.int32)
    n = np.array([10] * dim)
    N = np.prod(n)
    z = np.linspace(0, 2, num=n[0])
    x = onp.zeros((N, dim))
    for k in range(dim):
        nProd = int(np.prod(n[k + 1 :]))
        nStack = int(np.prod(n[0:k]))
        dark = np.hstack([z] * nProd)
        x[:, k] = onp.array([dark] * nStack).flatten()
    c = (1.0 + 1.0) / (x[-1, :] - x[0, :])
    z = (x - x[0, :]) * c + -1.0

    ncp1 = nCP(x[0, :], x[-1, :], nC, 5)
    ncp2 = nCP(x[0, :], x[-1, :], nC2, 10)
    Fc1 = ncp1.H(x.T, d, False)
    Fc2 = ncp2.H(x.T, d2, False)

    pncp1 = pnCP(x[0, :], x[-1, :], nC, 5)
    pncp2 = pnCP(x[0, :], x[-1, :], nC2, 10)
    Fp1 = pncp1.H(x.T, d, False)
    Fp2 = pncp2.H(x.T, d2, False)

    assert np.linalg.norm(Fc1 - Fp1, ord="fro") < 1e-14
    assert np.linalg.norm(Fc2 - Fp2, ord="fro") < 1e-14


def test_nLeP():
    from tfc.utils.BF.BF_Py import nLeP as pnLeP

    dim = 2
    nC = -1 * np.ones((dim, 1), dtype=np.int32)
    d = np.zeros(dim, dtype=np.int32)
    d2 = np.array([2, 3], dtype=np.int32)
    nC2 = np.block([[np.arange(4), -1.0 * np.ones(3)], [np.arange(7)]]).astype(np.int32)
    n = np.array([10] * dim)
    N = np.prod(n)
    z = np.linspace(0, 2, num=n[0])
    x = onp.zeros((N, dim))
    for k in range(dim):
        nProd = int(np.prod(n[k + 1 :]))
        nStack = int(np.prod(n[0:k]))
        dark = np.hstack([z] * nProd)
        x[:, k] = onp.array([dark] * nStack).flatten()
    c = (1.0 + 1.0) / (x[-1, :] - x[0, :])
    z = (x - x[0, :]) * c + -1.0

    nlep1 = nLeP(x[0, :], x[-1, :], nC, 5)
    nlep2 = nLeP(x[0, :], x[-1, :], nC2, 10)
    Fc1 = nlep1.H(x.T, d, False)
    Fc2 = nlep2.H(x.T, d2, False)

    pnlep1 = pnLeP(x[0, :], x[-1, :], nC, 5)
    pnlep2 = pnLeP(x[0, :], x[-1, :], nC2, 10)
    Fp1 = pnlep1.H(x.T, d, False)
    Fp2 = pnlep2.H(x.T, d2, False)

    assert np.linalg.norm(Fc1 - Fp1, ord="fro") < 1e-14
    assert np.linalg.norm(Fc2 - Fp2, ord="fro") < 1e-14


def test_nFS():
    from tfc.utils.BF.BF_Py import nFS as pnFS

    dim = 2
    nC = -1 * np.ones((dim, 1), dtype=np.int32)
    d = np.zeros(dim, dtype=np.int32)
    c = np.ones(dim)
    d2 = np.array([2, 3], dtype=np.int32)
    nC2 = np.block([[np.arange(4), -1.0 * np.ones(3)], [np.arange(7)]]).astype(np.int32)
    n = np.array([10] * dim)
    N = np.prod(n)
    z = np.linspace(0, 2.0 * np.pi, num=n[0])
    x = onp.zeros((N, dim))
    for k in range(dim):
        nProd = int(np.prod(n[k + 1 :]))
        nStack = int(np.prod(n[0:k]))
        dark = np.hstack([z] * nProd)
        x[:, k] = onp.array([dark] * nStack).flatten()
    c = (2.0 * np.pi) / (x[-1, :] - x[0, :])
    z = (x - x[0, :]) * c - np.pi

    nfs1 = nFS(x[0, :], x[-1, :], nC, 5)
    nfs2 = nFS(x[0, :], x[-1, :], nC2, 10)
    Fc1 = nfs1.H(x.T, d, False)
    Fc2 = nfs2.H(x.T, d2, False)

    pnfs1 = pnFS(x[0, :], x[-1, :], nC, 5)
    pnfs2 = pnFS(x[0, :], x[-1, :], nC2, 10)
    Fp1 = pnfs1.H(x.T, d, False)
    Fp2 = pnfs2.H(x.T, d2, False)

    assert np.linalg.norm(Fc1 - Fp1, ord="fro") < 1e-14
    assert np.linalg.norm(Fc2 - Fp2, ord="fro") < 5e-13


def test_nELMSigmoid():
    from tfc.utils.BF.BF_Py import nELMSigmoid as pnELMSigmoid

    dim = 2
    nC = -1 * np.ones(1, dtype=np.int32)
    d = np.zeros(dim, dtype=np.int32)
    d2 = np.array([2, 3], dtype=np.int32)
    nC2 = np.array([4], dtype=np.int32)
    n = np.array([10] * dim)
    N = np.prod(n)
    z = np.linspace(0, 1, num=n[0])
    X = onp.zeros((N, dim))
    for k in range(dim):
        nProd = int(np.prod(n[k + 1 :]))
        nStack = int(np.prod(n[0:k]))
        dark = np.hstack([z] * nProd)
        X[:, k] = onp.array([dark] * nStack).flatten()
    c = 1.0 / (X[-1, :] - X[0, :])
    z = (X - X[0, :]) * c

    elm1 = nELMSigmoid(X[0, :], X[-1, :], nC, 10)
    w = elm1.w
    b = elm1.b
    elm2 = nELMSigmoid(X[0, :], X[-1, :], nC2, 10)
    elm2.w = w
    elm2.b = b
    Fc1 = elm1.H(X.T, d, False)
    Fc2 = elm2.H(X.T, d2, False)

    pelm1 = pnELMSigmoid(X[0, :], X[-1, :], nC, 10)
    pelm1.w = w
    pelm1.b = b
    pelm2 = pnELMSigmoid(X[0, :], X[-1, :], nC2, 10)
    pelm2.w = w
    pelm2.b = b
    Fp1 = pelm1.H(X.T, d, False)
    Fp2 = pelm2.H(X.T, d2, False)

    assert np.linalg.norm(Fc1 - Fp1, ord="fro") < 1e-14
    assert np.linalg.norm(Fc2 - Fp2, ord="fro") < 1e-13


def test_nELMTanh():
    from tfc.utils.BF.BF_Py import nELMTanh as pnELMTanh

    dim = 2
    nC = -1 * np.ones(1, dtype=np.int32)
    d = np.zeros(dim, dtype=np.int32)
    c = np.ones(dim)
    d2 = np.array([2, 3], dtype=np.int32)
    nC2 = np.array([4], dtype=np.int32)
    n = np.array([10] * dim)
    N = np.prod(n)
    z = np.linspace(0, 1, num=n[0])
    X = onp.zeros((N, dim))
    for k in range(dim):
        nProd = int(np.prod(n[k + 1 :]))
        nStack = int(np.prod(n[0:k]))
        dark = np.hstack([z] * nProd)
        X[:, k] = onp.array([dark] * nStack).flatten()
    c = 1.0 / (X[-1, :] - X[0, :])
    z = (X - X[0, :]) * c

    elm1 = nELMTanh(X[0, :], X[-1, :], nC, 10)
    w = elm1.w
    b = elm1.b
    elm2 = nELMTanh(X[0, :], X[-1, :], nC2, 10)
    elm2.w = w
    elm2.b = b
    Fc1 = elm1.H(X.T, d, False)
    Fc2 = elm2.H(X.T, d2, False)

    pelm1 = pnELMTanh(X[0, :], X[-1, :], nC, 10)
    pelm1.w = w
    pelm1.b = b
    pelm2 = pnELMTanh(X[0, :], X[-1, :], nC2, 10)
    pelm2.w = w
    pelm2.b = b
    Fp1 = pelm1.H(X.T, d, False)
    Fp2 = pelm2.H(X.T, d2, False)

    assert np.linalg.norm(Fc1 - Fp1, ord="fro") < 1e-14
    assert np.linalg.norm(Fc2 - Fp2, ord="fro") < 1e-13


def test_nELMSin():
    from tfc.utils.BF.BF_Py import nELMSin as pnELMSin

    dim = 2
    nC = -1 * np.ones(1, dtype=np.int32)
    d = np.zeros(dim, dtype=np.int32)
    c = np.ones(dim)
    d2 = np.array([2, 3], dtype=np.int32)
    nC2 = np.array([4], dtype=np.int32)
    n = np.array([10] * dim)
    N = np.prod(n)
    z = np.linspace(0, 1, num=n[0])
    X = onp.zeros((N, dim))
    for k in range(dim):
        nProd = int(np.prod(n[k + 1 :]))
        nStack = int(np.prod(n[0:k]))
        dark = np.hstack([z] * nProd)
        X[:, k] = onp.array([dark] * nStack).flatten()
    c = 1.0 / (X[-1, :] - X[0, :])
    z = (X - X[0, :]) * c

    elm1 = nELMSin(X[0, :], X[-1, :], nC, 10)
    w = elm1.w
    b = elm1.b
    elm2 = nELMSin(X[0, :], X[-1, :], nC2, 10)
    elm2.w = w
    elm2.b = b
    Fc1 = elm1.H(X.T, d, False)
    Fc2 = elm2.H(X.T, d2, False)

    pelm1 = pnELMSin(X[0, :], X[-1, :], nC, 10)
    pelm1.w = w
    pelm1.b = b
    pelm2 = pnELMSin(X[0, :], X[-1, :], nC2, 10)
    pelm2.w = w
    pelm2.b = b

    Fp1 = pelm1.H(X.T, d, False)
    Fp2 = pelm2.H(X.T, d2, False)

    assert np.linalg.norm(Fc1 - Fp1, ord="fro") < 1e-14
    assert np.linalg.norm(Fc2 - Fp2, ord="fro") < 1e-12


def test_nELMSwish():
    from tfc.utils.BF.BF_Py import nELMSwish as pnELMSwish

    dim = 2
    nC = -1 * np.ones(1, dtype=np.int32)
    d = np.zeros(dim, dtype=np.int32)
    c = np.ones(dim)
    d2 = np.array([2, 3], dtype=np.int32)
    nC2 = np.array([4], dtype=np.int32)
    n = np.array([10] * dim)
    N = np.prod(n)
    z = np.linspace(0, 1, num=n[0])
    X = onp.zeros((N, dim))
    for k in range(dim):
        nProd = int(np.prod(n[k + 1 :]))
        nStack = int(np.prod(n[0:k]))
        dark = np.hstack([z] * nProd)
        X[:, k] = onp.array([dark] * nStack).flatten()
    c = 1.0 / (X[-1, :] - X[0, :])
    z = (X - X[0, :]) * c

    elm1 = nELMSwish(X[0, :], X[-1, :], nC, 10)
    w = elm1.w
    b = elm1.b
    elm2 = nELMSwish(X[0, :], X[-1, :], nC2, 10)
    elm2.w = w
    elm2.b = b
    Fc1 = elm1.H(X.T, d, False)
    Fc2 = elm2.H(X.T, d2, False)

    pelm1 = pnELMSwish(X[0, :], X[-1, :], nC, 10)
    pelm1.w = w
    pelm1.b = b
    pelm2 = pnELMSwish(X[0, :], X[-1, :], nC2, 10)
    pelm2.w = w
    pelm2.b = b
    Fp1 = pelm1.H(X.T, d, False)
    Fp2 = pelm2.H(X.T, d2, False)

    assert np.linalg.norm(Fc1 - Fp1, ord="fro") < 1e-14
    assert np.linalg.norm(Fc2 - Fp2, ord="fro") < 1e-12


def test_nELMReLU():
    from tfc.utils.BF.BF_Py import nELMReLU as pnELMReLU

    dim = 2
    nC = -1 * np.ones(1, dtype=np.int32)
    d = np.zeros(dim, dtype=np.int32)
    c = np.ones(dim)
    d2 = np.array([0, 1], dtype=np.int32)
    d3 = np.array([1, 1], dtype=np.int32)
    d4 = np.array([0, 2], dtype=np.int32)
    nC2 = np.array([4], dtype=np.int32)
    n = np.array([10] * dim)
    N = np.prod(n)
    z = np.linspace(0, 1, num=n[0])
    X = onp.zeros((N, dim))
    for k in range(dim):
        nProd = int(np.prod(n[k + 1 :]))
        nStack = int(np.prod(n[0:k]))
        dark = np.hstack([z] * nProd)
        X[:, k] = onp.array([dark] * nStack).flatten()
    c = 1.0 / (X[-1, :] - X[0, :])
    z = (X - X[0, :]) * c

    elm1 = nELMReLU(X[0, :], X[-1, :], nC, 10)
    w = elm1.w
    b = elm1.b
    elm2 = nELMReLU(X[0, :], X[-1, :], nC2, 10)
    elm2.w = w
    elm2.b = b
    Fc1 = elm1.H(X.T, d, False)
    Fc2 = elm2.H(X.T, d2, False)
    Fc3 = elm2.H(X.T, d3, False)
    Fc4 = elm2.H(X.T, d4, False)

    pelm1 = pnELMReLU(X[0, :], X[-1, :], nC, 10)
    pelm1.w = w
    pelm1.b = b
    pelm2 = pnELMReLU(X[0, :], X[-1, :], nC2, 10)
    pelm2.w = w
    pelm2.b = b
    Fp1 = pelm1.H(X.T, d, False)
    Fp2 = pelm2.H(X.T, d2, False)
    Fp3 = pelm2.H(X.T, d3, False)
    Fp4 = pelm2.H(X.T, d4, False)

    assert np.linalg.norm(Fc1 - Fp1, ord="fro") < 1e-14
    assert np.linalg.norm(Fc2 - Fp2, ord="fro") < 1e-14
    assert np.linalg.norm(Fc3 - Fp3, ord="fro") < 1e-14
    assert np.linalg.norm(Fc4 - Fp4, ord="fro") < 1e-14
