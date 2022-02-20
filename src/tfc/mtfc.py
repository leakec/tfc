from jax.config import config

config.update("jax_enable_x64", True)

from copy import copy
import numpy as onp
import jax.numpy as np
from jax import core, abstract_arrays, jvp
from jax.interpreters import ad, batching, xla
from jax.ops import index_update, index
from jax.lib import xla_client

from .utils.TFCUtils import TFCPrint


class mtfc:
    """
    This is the multivariate TFC class. It acts as a container that holds:

    * The linear map from the domain of the DE to the domain of the free-function.
    * The necessary JAX code that enables automatic differentiation of the constrained experssion and Jacobians of the residual with respect to the unknown coefficients in the linear combination of basis functions that make up the free function.
    * Other useful TFC related functions such as collocation point creation.

    In addition, this class ties these methods together to form a utility that enables a higher level of code abstraction
    such that the end-user scripts are simple, clear, and elegant implementations of TFC.

    Parameters
    ----------
    n : list or array-like
        Number of points to use per-dimension when discretizing the domain. List or array must be same lenght as number of dimensions.

    nC : int or list or array-like
        Number of functions to remove from the beginning of free function linear expansion. This variable is used to account for basis functions that are linearly dependent on support functions used in the construction of the constrained expressions. The constraints for each dimension can be expressed in 1 of 2 ways. Note that a value of -1 is used to indicate no constraints exist for a particular dimension.

        1. As an integer. When expressed as an integer, the first nC basis functions are removed from the free function.
        2. As a list or array. When expressed as a list or array, the basis functions corresponding to the numbers given by the list or array are removed from the free function.

    deg : int
        Degree of the basis function expansion.

    basis : {"CP","LeP","FS","ELMTanh","ELMSigmoid","ELMSin","ELMSwish","ELMReLU"}, optional
        This optional keyword argument specifies the basis functions to be used. (Default value = "CP")

    x0 : list or array-like
        Specifies the beginning of the DE domain. (Default value = None)

    xf : list or array-like
        Specifies the end of the DE domain. (Default value = None)

    """

    def __init__(self, n, nC, deg, dim=2, basis="CP", x0=None, xf=None):

        # Store givens
        self._elm_classes = ["ELMSigmoid", "ELMTanh", "ELMSin", "ELMSwish", "ELMReLU"]
        self.deg = deg
        self.dim = dim

        # Set N based on user input
        if isinstance(n, np.ndarray):
            if not n.flatten().shape[0] == dim:
                TFCPrint.Error(
                    "n has length "
                    + str(n.flatten().shape[0])
                    + ", but it should be equal to the number of dimensions, "
                    + str(dim)
                    + "."
                )
            self.n = n.astype(np.int32)
        else:
            if not len(n) == dim:
                TFCPrint.Error(
                    "n has length "
                    + str(n)
                    + ", but it should be equal to the number of dimensions, "
                    + str(dim)
                    + "."
                )
            self.n = np.array(n, dtype=np.int32)
        self.N = int(np.prod(self.n, dtype=np.int32))

        self.basis = basis

        # Set x0 based on user input
        if x0 is None:
            self.x0 = np.zeros(dim)
        else:
            if isinstance(x0, np.ndarray):
                if not x0.flatten().shape[0] == dim:
                    TFCPrint.Error(
                        "x0 has length "
                        + str(x0.flatten().shape[0])
                        + ", but it should be equal to the number of dimensions, "
                        + str(dim)
                        + "."
                    )
                self.x0 = x0
            else:
                if not len(x0) == dim:
                    TFCPrint.Error(
                        "x0 has length "
                        + len(x0)
                        + ", but it should be equal to the number of dimensions, "
                        + str(dim)
                        + "."
                    )
                self.x0 = np.array(x0).flatten()
                if not self.x0.shape[0] == dim:
                    TFCPrint.Error(
                        "x0 has length "
                        + str(x0.flatten().shape[0])
                        + ", but it should be equal to the number of dimensions, "
                        + str(dim)
                        + "."
                    )

        # Set xf based on user input
        if xf is None:
            self.xf = np.zeros(dim)
        else:
            if isinstance(xf, np.ndarray):
                if not xf.flatten().shape[0] == dim:
                    TFCPrint.Error(
                        "xf has length "
                        + str(xf.flatten().shape[0])
                        + ", but it should be equal to the number of dimensions, "
                        + str(dim)
                        + "."
                    )
                self.xf = xf
            else:
                if not len(xf) == dim:
                    TFCPrint.Error(
                        "xf has length "
                        + len(xf)
                        + ", but it should be equal to the number of dimensions, "
                        + str(dim)
                        + "."
                    )
                self.xf = np.array(xf).flatten()
                if not self.xf.shape[0] == dim:
                    TFCPrint.Error(
                        "xf has length "
                        + str(xf.flatten().shape[0])
                        + ", but it should be equal to the number of dimensions, "
                        + str(dim)
                        + "."
                    )

        # Create nC matrix based on user input
        if basis in self._elm_classes:
            if isinstance(nC, int):
                self.nC = onp.arange(nC, dtype=onp.int32)
            elif isinstance(nC, np.ndarray):
                self.nC = nC.astype(onp.int32)
            elif isinstance(nC, list):
                self.nC = np.array(nC, dtype=np.int32)
            if self.nC.shape[0] > self.deg:
                TFCPrint.Error("Number of basis functions is less than number of constraints!")
            if np.any(self.nC < 0):
                TFCPrint.Error(
                    "To set nC to -1 (no constraints) either use nC = -1 or nC = 0 (i.e., use an integer not a list or array). Do not put only -1 in a list or array, this will cause issues in the C++ layer."
                )
        else:
            if isinstance(nC, np.ndarray) and len(nC.shape) > 1:
                if not nC.shape[0] == self.dim:
                    TFCPrint.Error(
                        "nC has "
                        + str(nC.flatten().shape[0])
                        + " rows, but the row number should be equal to the number of dimensions, "
                        + str(dim)
                        + "."
                    )
                self.nC = nC.astype(np.int32)
            else:
                if isinstance(nC, np.ndarray):
                    nC = nC.tolist()
                if not len(nC) == dim:
                    TFCPrint.Error(
                        "nC has length "
                        + str(len(nC))
                        + ", but it should be equal to the number of dimensions, "
                        + str(dim)
                        + "."
                    )
                nCmax = 0
                for k in range(dim):
                    if isinstance(nC[k], np.ndarray):
                        nCk = np.array(nC[k]).flatten()
                    else:
                        nCk = np.array([nC[k]]).flatten()
                    if nCk.shape[0] == 1:
                        maxk = nCk[0]
                    else:
                        maxk = nCk.shape[0]
                    if maxk > nCmax:
                        nCmax = maxk

                if nCmax > self.deg:
                    TFCPrint.Error(
                        "Number of basis functions is less than the number of constraints!"
                    )

                onC = onp.zeros((dim, nCmax))
                for k in range(dim):
                    if isinstance(nC[k], np.ndarray):
                        nCk = np.array(nC[k]).flatten()
                    else:
                        nCk = onp.array([nC[k]]).flatten()
                    n = nCk.shape[0]
                    if n == 1:
                        nCk = onp.arange(nCk[0])
                        n = nCk.shape[0]
                    if n < nCmax:
                        if n == 0:
                            nCk = -1.0 * onp.ones(nCmax)
                        else:
                            nCk = np.hstack([nCk, -1 * np.ones(nCmax - n)])
                    onC[k, :] = nCk.astype(np.int32)
                self.nC = np.array(onC.tolist(), dtype=np.int32)

        # Setup the basis function
        if self.basis == "CP":
            from .utils.BF import nCP

            self.basisClass = nCP(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = -1.0
            zf = 1.0
        elif self.basis == "LeP":
            from .utils.BF import nLeP

            self.basisClass = nLeP(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = -1.0
            zf = 1.0
        elif self.basis == "FS":
            from .utils.BF import nFS

            self.basisClass = nFS(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = -np.pi
            zf = np.pi
        elif self.basis == "ELMSigmoid":
            from .utils.BF import nELMSigmoid

            self.basisClass = nELMSigmoid(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = 0.0
            zf = 1.0
        elif self.basis == "ELMTanh":
            from .utils.BF import nELMTanh

            self.basisClass = nELMTanh(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = 0.0
            zf = 1.0
        elif self.basis == "ELMSin":
            from .utils.BF import nELMSin

            self.basisClass = nELMSin(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = 0.0
            zf = 1.0
        elif self.basis == "ELMSwish":
            from .utils.BF import nELMSwish

            self.basisClass = nELMSwish(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = 0.0
            zf = 1.0
        elif self.basis == "ELMReLU":
            from .utils.BF import nELMReLU

            self.basisClass = nELMReLU(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = 0.0
            zf = 1.0
        else:
            TFCPrint.Error("Invalid basis selection. Please select a valid basis")

        if self.basisClass.numBasisFunc > self.N:
            TFCPrint.Warning(
                "Warning, you have more basis functions than points!\nThis may lead to large solution errors!"
            )

        self.c = self.basisClass.c

        # Calculate z points and corresponding x
        self.z = onp.zeros((self.dim, self.N))
        x = tuple([onp.zeros(self.N) for x in range(self.dim)])
        if self.basis in ["CP", "LeP"]:
            for k in range(self.dim):
                nProd = onp.prod(self.n[k + 1 :])
                nStack = onp.prod(self.n[0:k])
                n = self.n[k] - 1
                I = onp.linspace(0, n, n + 1).reshape((n + 1, 1))
                dark = onp.cos(np.pi * (n - I) / float(n))
                dark = onp.hstack([dark] * nProd).flatten()
                self.z[k, :] = onp.array([dark] * nStack).flatten()
                x[k][:] = (self.z[k, :] - z0) / self.c[k] + self.x0[k]
        else:
            for k in range(self.dim):
                nProd = onp.prod(self.n[k + 1 :])
                nStack = onp.prod(self.n[0:k])
                dark = onp.linspace(z0, zf, num=self.n[k]).reshape((self.n[k], 1))
                dark = onp.hstack([dark] * nProd).flatten()
                self.z[k, :] = onp.array([dark] * nStack).flatten()
                x[k][:] = (self.z[k, :] - z0) / self.c[k] + self.x0[k]

        self.z = np.array(self.z.tolist())
        self.x = tuple([np.array(x[k].tolist()) for k in range(self.dim)])

        self.SetupJAX()

    def H(self, *x, full=False):
        """
        This function computes the basis function matrix for the points specified by *x.

        Parameters
        ----------
        *x : iterable of array-like
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        H : array-like
            Basis function matrix.
        """
        d = onp.zeros(self.dim, dtype=np.int32)
        return self._Hjax(*x, d=d, full=full)

    def Hx(self, *x, full=False):
        """
        This function computes the derivative of the basis function matrix for the points specified by *x with respect to the first variable.

        Parameters
        ----------
        *x : iterable of array-like
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        Hx : array-like
            Derivative of the basis function matrix with respect to the first variable.
        """
        d = onp.zeros(self.dim, dtype=np.int32)
        d[0] = 1
        return self._Hjax(*x, d=d, full=full)

    def Hx2(self, *x, full=False):
        """
        This function computes the second derivative of the basis function matrix for the points specified by *x with respect to the first variable.

        Parameters
        ----------
        *x : iterable of array-like
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        Hx2 : array-like
            Second derivative of the basis function matrix with respect to the first variable.
        """
        d = onp.zeros(self.dim, dtype=np.int32)
        d[0] = 2
        return self._Hjax(*x, d=d, full=full)

    def Hy2(self, *x, full=False):
        """
        This function computes the second derivative of the basis function matrix for the points specified by *x with respect to the second variable.

        Parameters
        ----------
        *x : iterable of array-like
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        Hy2 : array-like
            Second derivative of the basis function matrix with respect to the second variable.
        """
        d = onp.zeros(self.dim, dtype=np.int32)
        d[1] = 2
        return self._Hjax(*x, d=d, full=full)

    def Hx2y(self, *x, full=False):
        """
        This function computes the mixed derivative (second order derivative with respect to the first variable and first order with respect
        to the second variable) of the basis function matrix for the points specified by *x.

        Parameters
        ----------
        *x : iterable of array-like
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        Hx2y : array-like
            Mixed derivative of the basis function matrix with respect to the first variable.
        """
        d = onp.zeros(self.dim, dtype=np.int32)
        d[0] = 2
        d[1] = 1
        return self._Hjax(*x, d=d, full=full)

    def Hy(self, *x, full=False):
        """
        This function computes the derivative of the basis function matrix for the points specified by *x with respect to the second variable.

        Parameters
        ----------
        *x : iterable of array-like
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        Hy : array-like
            Derivative of the basis function matrix with respect to the second variable.
        """
        d = onp.zeros(self.dim, dtype=np.int32)
        d[1] = 1
        return self._Hjax(*x, d=d, full=full)

    def Hxy(self, *x, full=False):
        """
        This function computes the mixed derivative (first order derivative with respect to the first variable and first order with respect
        to the second variable) of the basis function matrix for the points specified by *x.

        Parameters
        ----------
        *x : iterable of array-like
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        Hxy : array-like
            Mixed derivative of the basis function matrix with respect to the first variable.
        """
        d = onp.zeros(self.dim, dtype=np.int32)
        d[0] = 1
        d[1] = 1
        return self._Hjax(*x, d=d, full=full)

    def Hz(self, *x, full=False):
        """
        This function computes the derivative of the basis function matrix for the points specified by *x with respect to the third variable.

        Parameters
        ----------
        *x : iterable of array-like
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        Hz : array-like
            Derivative of the basis function matrix with respect to the third variable.
        """
        d = onp.zeros(self.dim, dtype=np.int32)
        d[2] = 1
        return self._Hjax(*x, d=d, full=full)

    def SetupJAX(self):
        """This function is used internally by TFC to setup autograd primatives and create desired behavior when taking derivatives of TFC constrained expressions."""

        # Helper functions
        def _constant_bool(c, a):
            return xla_client.ops.Constant(c, bool(a))

        def _constant_s32_scalar(c, a):
            return xla_client.ops.Constant(c, int(a))

        def _constant_array(c, a):
            return xla_client.ops.Constant(c, a)

        def _unpack_builder(c):
            # If `c` is a ComputationBuilder object, extracts the underlying XlaBuilder.
            return getattr(c, "_builder", c)

        d0 = onp.zeros(self.dim, dtype=np.int32)

        # Regiser XLA function
        obj = self.basisClass.xlaCapsule
        xlaName = "BasisFunc" + str(self.basisClass.identifier)
        xlaName = xlaName.encode("utf-8")
        xla_client.register_custom_call_target(xlaName, obj, platform="cpu")

        # Create Primitives
        H_p = core.Primitive("H")

        def Hjax(*x, d=d0, full=False):
            return H_p.bind(*x, d=d, full=full)

        # Implicit translations
        def H_impl(*x, d=d0, full=False):
            return self.basisClass.H(np.array(x), d, full)

        H_p.def_impl(H_impl)

        # Define abstract evaluation
        def H_abstract_eval(*x, d=d0, full=False):
            if full:
                dim1 = self.basisClass.numBasisFuncFull
            else:
                dim1 = self.basisClass.numBasisFunc
            if len(x[0].shape) == 0:
                dims = (dim1,)
            else:
                dims = (x[0].shape[0], dim1)
            return abstract_arrays.ShapedArray(dims, x[0].dtype)

        H_p.def_abstract_eval(H_abstract_eval)

        # XLA compilation
        def H_xla(c, *x, d=d0, full=False):
            c = _unpack_builder(c)
            x_shape = c.get_shape(x[0])
            dims = x_shape.dimensions()
            dtype = x_shape.element_type()
            dim0 = dims[0]
            if full:
                dim1 = self.basisClass.numBasisFuncFull
            else:
                dim1 = self.basisClass.numBasisFunc
            return xla_client.ops.CustomCall(
                c,
                xlaName,
                (
                    _constant_s32_scalar(c, self.basisClass.identifier),
                    xla_client.ops.ConcatInDim(c, x, 0),
                    _constant_array(c, d),
                    _constant_s32_scalar(c, self.dim),
                    _constant_bool(c, full),
                    _constant_s32_scalar(c, dim0),
                    _constant_s32_scalar(c, dim1),
                ),
                xla_client.Shape.array_shape(dtype, (dim0, dim1)),
            )

        xla.backend_specific_translations["cpu"][H_p] = H_xla

        # Batching translation
        def H_batch(vec, batch, d=d0, full=False):
            return Hjax(*vec, d=d, full=full), batch[0]

        batching.primitive_batchers[H_p] = H_batch

        # Jacobian vector translation
        def H_jvp(arg_vals, arg_tans, d=d0, full=False):
            n = len(arg_vals)
            flat = len(arg_vals[0].shape) == 1
            dim0 = arg_vals[0].shape[0]
            if full:
                dim1 = self.basisClass.numBasisFuncFull
            else:
                dim1 = self.basisClass.numBasisFunc
            out_tans = np.zeros((dim0, dim1))
            for k in range(n):
                if not (type(arg_tans[k]) is ad.Zero):
                    if type(arg_tans[k]) is batching.BatchTracer:
                        flag = onp.any(arg_tans[k].val != 0)
                    else:
                        flag = onp.any(arg_tans[k] != 0)
                    if flag:
                        dark = copy(d)
                        dark[k] += 1
                        if flat:
                            out_tans += Hjax(*arg_vals, d=dark, full=full) * np.expand_dims(
                                arg_tans[k], 1
                            )
                        else:
                            out_tans += Hjax(*arg_vals, d=dark, full=full) * arg_tans[k]
            return (Hjax(*arg_vals, d=d, full=full), out_tans)

        ad.primitive_jvps[H_p] = H_jvp

        self._Hjax = Hjax
