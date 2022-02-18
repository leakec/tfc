from jax.config import config

config.update("jax_enable_x64", True)

import numpy as onp
import jax.numpy as np
from jax import core, abstract_arrays, jvp
from jax.interpreters import ad, batching, xla
from jax.lib import xla_client

from .utils.TFCUtils import TFCPrint


class utfc:
    """
    This is the univariate TFC class. It acts as a container that creates and stores:

    * The linear map between the free function domain (z) and the problem domain (x).
    * The basis functions or ELMs that make up the free function.
    * The necessary JAX code that enables automatic differentiation of the free function.
    * Other useful TFC related functions such as collocation point creation.

    In addition, this class ties these methods together to form a utility that enables a higher level of code abstraction
    such that the end-user scripts are simple, clear, and elegant implementations of TFC.

    Parameters
    ----------
    N : int
        Number of points to use when discretizing the domain.
    nC : int or list or array-like
        Number of functions to remove from the free function linear expansion. This variable is used to account for basis functions that are linearly dependent on support functions used in the construction of the constrained expressions. It can be expressed in 1 of 2 ways.

        1. As an integer. When expressed as an integer, the first nC basis functions are removed from the free function.
        2. As a list or array. When expressed as a list or array, the basis functions corresponding to the numbers given by the list or array are removed from the free function.

    m : int
        Degree of the basis function expansion. This number is one less than the number of basis functions used before removing those specified by nC.
    x0 : float, optional
        Specifies the beginning of the DE domain. (Default value = 0)
    xf : float
        This required keyword argument specifies the end of the DE domain.
    basis : {"CP","LeP","FS","ELMTanh","ELMSigmoid","ELMSin","ELMSwish","ELMReLU"}, optional
        This optional keyword argument specifies the basis functions to be used. (Default value = "CP")
    """

    def __init__(self, N, nC, deg, basis="CP", x0=None, xf=None):
        """
        Constructor for the utfc class.

        Parameters
        ----------
        N : int
            Number of points to use when discretizing the domain.
        nC : int or list or array-like
            Number of functions to remove from the free function linear expansion. This variable is used to account for basis functions that are linearly dependent on support functions used in the construction of the constrained expressions. It can be expressed in 1 of 2 ways.
            1. As an integer. When expressed as an integer, the first nC basis functions are removed from the free function.
            2. As a list or array. When expressed as a list or array, the basis functions corresponding to the numbers given by the list or array are removed from the free function.
        m : int
            Degree of the basis function expansion. This number is one less than the number of basis functions used before removing those specified by nC.
        x0 : float, optional
            Specifies the beginning of the DE domain. (Default value = 0)
        xf : float
            This required keyword argument specifies the end of the DE domain.
        basis : {"CP","LeP","FS","ELMTanh","ELMSigmoid","ELMSin","ELMSwish","ELMReLU"}, optional
            This optional keyword argument specifies the basis functions to be used. (Default value = "CP")
        """

        # Store givens
        self.N = N
        self.deg = deg

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

        self.basis = basis

        if x0 is None:
            self.x0 = 0.0
        else:
            self.x0 = x0

        if xf is None:
            self.xf = 0.0
        else:
            self.xf = xf

        # Setup the basis function
        if self.basis == "CP":
            from .utils.BF import CP

            self.basisClass = CP(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = -1.0
            zf = 1.0
        elif self.basis == "LeP":
            from .utils.BF import LeP

            self.basisClass = LeP(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = -1.0
            zf = 1.0
        elif self.basis == "FS":
            from .utils.BF import FS

            self.basisClass = FS(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = -np.pi
            zf = np.pi
        elif self.basis == "ELMReLU":
            from .utils.BF import ELMReLU

            self.basisClass = ELMReLU(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = 0.0
            zf = 1.0
        elif self.basis == "ELMSigmoid":
            from .utils.BF import ELMSigmoid

            self.basisClass = ELMSigmoid(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = 0.0
            zf = 1.0
        elif self.basis == "ELMTanh":
            from .utils.BF import ELMTanh

            self.basisClass = ELMTanh(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = 0.0
            zf = 1.0
        elif self.basis == "ELMSin":
            from .utils.BF import ELMSin

            self.basisClass = ELMSin(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = 0.0
            zf = 1.0
        elif self.basis == "ELMSwish":
            from .utils.BF import ELMSwish

            self.basisClass = ELMSwish(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = 0.0
            zf = 1.0
        else:
            TFCPrint.Error("Invalid basis selection. Please select a valid basis")

        self.c = self.basisClass.c

        # Calculate z points and corresponding x
        if self.basis in ["CP", "LeP"]:
            n = self.N - 1
            I = np.linspace(0, n, n + 1)
            self.z = np.cos(np.pi * (n - I) / float(n))
            self.x = (self.z - z0) / self.c + self.x0
        else:
            self.z = np.linspace(z0, zf, self.N)
            self.x = (self.z - z0) / self.c + self.x0

        self._SetupJax()

    def H(self, x, full=False):
        """
        This function computes the basis function matrix for the points specified by x.

        Parameters
        ----------
        x : array-like
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        H : array-like
            Basis function matrix.
        """
        return self._Hjax(x, d=0, full=full)

    def dH(self, x, full=False):
        """This function computes the deriative of H. See documentation of 'H' for more details.

        Parameters
        ----------
        x : array-like
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        H : array-like
            Derivative of the basis function matrix.
        """
        return self._Hjax(x, d=1, full=full)

    def d2H(self, x, full=False):
        """This function computes the second deriative of H. See documentation of H for more details.

        Parameters
        ----------
        x : array-like
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        d2H : array-like
            Second derivative of the basis function matrix.
        """
        return self._Hjax(x, d=2, full=full)

    def d4H(self, x, full=False):
        """This function computes the fourth deriative of H. See documentation of H for more details.

        Parameters
        ----------
        x : array-like
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        d4H : array-like
            Fourth derivative of the basis function matrix.
        """
        return self._Hjax(x, d=4, full=full)

    def d8H(self, x, full=False):
        """This function computes the eighth deriative of H. See documentation of H for more details.

        Parameters
        ----------
        x : array-like
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        d8H : array-like
            Eighth derivative of the basis function matrix.
        """
        return self._Hjax(x, d=8, full=full)

    def _SetupJax(self):
        """This function is used internally by TFC to setup JAX primatives and create desired behavior when taking derivatives of TFC constrained expressions."""

        # Helper functions
        def _constant_bool(c, a):
            return xla_client.ops.Constant(c, bool(a))

        def _constant_s32_scalar(c, a):
            return xla_client.ops.Constant(c, int(a))

        def _unpack_builder(c):
            # If `c` is a ComputationBuilder object, extracts the underlying XlaBuilder.
            return getattr(c, "_builder", c)

        # Regiser XLA function
        obj = self.basisClass.xlaCapsule
        xlaName = "BasisFunc" + str(self.basisClass.identifier)
        xlaName = xlaName.encode("utf-8")
        xla_client.register_custom_call_target(xlaName, obj, platform="cpu")

        # Create primitives
        H_p = core.Primitive("H")

        def Hjax(x, d=0, full=False):
            return H_p.bind(x, d=d, full=full)

        # Implicit translation
        def H_impl(x, d=0, full=False):
            return self.basisClass.H(x, d, full)

        H_p.def_impl(H_impl)

        # Abstract evaluation
        def H_abstract_eval(x, d=0, full=False):
            if full:
                dim1 = self.basisClass.m
            else:
                dim1 = self.basisClass.m - self.basisClass.numC
            if len(x.shape) == 0:
                dims = (dim1,)
            else:
                dims = (x.shape[0], dim1)
            return abstract_arrays.ShapedArray(dims, x.dtype)

        H_p.def_abstract_eval(H_abstract_eval)

        # XLA compilation
        def H_xla(c, x, d=0, full=False):
            c = _unpack_builder(c)
            x_shape = c.get_shape(x)
            dims = x_shape.dimensions()
            dtype = x_shape.element_type()
            dim0 = dims[0]
            if full:
                dim1 = self.basisClass.m
            else:
                dim1 = self.basisClass.m - self.basisClass.numC
            return xla_client.ops.CustomCall(
                c,
                xlaName,
                (
                    _constant_s32_scalar(c, self.basisClass.identifier),
                    x,
                    _constant_s32_scalar(c, d),
                    _constant_bool(c, full),
                    _constant_s32_scalar(c, dim0),
                    _constant_s32_scalar(c, dim1),
                ),
                xla_client.Shape.array_shape(dtype, (dim0, dim1)),
            )

        xla.backend_specific_translations["cpu"][H_p] = H_xla

        # Define batching translation
        def H_batch(vec, batch, d=0, full=False):
            return Hjax(*vec, d=d, full=full), batch[0]

        batching.primitive_batchers[H_p] = H_batch

        # Define jacobain vector product
        def H_jvp(arg_vals, arg_tans, d=0, full=False):
            x = arg_vals[0]
            dx = arg_tans[0]
            if not (dx is ad.Zero):
                if type(dx) is batching.BatchTracer:
                    flag = onp.any(dx.val != 0)
                else:
                    flag = onp.any(dx != 0)
                if flag:
                    if len(dx.shape) == 1:
                        out_tans = Hjax(x, d=d + 1, full=full) * onp.expand_dims(dx, 1)
                    else:
                        out_tans = Hjax(x, d=d + 1, full=full) * dx
            else:
                dim0 = x.shape[0]
                if full:
                    dim1 = self.basisClass.m
                else:
                    dim1 = self.basisClass.m - self.basisClass.numC
                out_tans = np.zeros((dim0, dim1))
            return (Hjax(x, d=d, full=full), out_tans)

        ad.primitive_jvps[H_p] = H_jvp

        # Provide pointer for TFC class
        self._Hjax = Hjax


class HybridUtfc:
    """
    This class combines TFC classes together so that multiple basis functions can be used
    simultaneously in the solution. Note, that this class is not yet complete.

    Parameters
    ----------
    tfcClasses : list of utfc classes
        This list of utfc classes make up the basis functions used in the HybridUtfc class.
    """

    def __init__(self, tfcClasses):
        """
        This function computes the basis function matrix for the points specified by x.

        Parameters
        ----------
        tfcClasses : list of utfc classes
            This list of utfc classes make up the basis functions used in the HybridUtfc class.
        """

        if not all([k.N == tfcClasses[0].N for k in tfcClasses]):
            TFCPrint.Error("Not all TFC classes provided have the same number of points.")
        self._tfcClasses = tfcClasses

    def H(self, x, full=False):
        """
        This function computes the basis function matrix for the points specified by x.

        Parameters
        ----------
        x : array-like
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        H : array-like
            Basis function matrix.
        """
        return np.hstack([k._Hjax(x, d=0, full=full) for j, k in enumerate(self._tfcClasses)])

    def dH(self, x, full=False):
        """
        This function computes the derivative of the basis function matrix for the points specified by x.

        Parameters
        ----------
        x : array-like
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        dH : array-like
            Derivative of the basis function matrix.
        """
        return np.hstack([k._Hjax(x, d=1, full=full) for j, k in enumerate(self._tfcClasses)])

    def d2H(self, x, full=False):
        """
        This function computes the second derivative of the basis function matrix for the points specified by x.

        Parameters
        ----------
        x : array-like
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        d2H : array-like
            Second derivative of the basis function matrix.
        """
        return np.hstack([k._Hjax(x, d=2, full=full) for j, k in enumerate(self._tfcClasses)])

    def d3H(self, x, full=False):
        """
        This function computes the third derivative of the basis function matrix for the points specified by x.

        Parameters
        ----------
        x : array-like
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        d3H : array-like
            Third derivative of the basis function matrix.
        """
        return np.hstack([k._Hjax(x, d=3, full=full) for j, k in enumerate(self._tfcClasses)])

    def d4H(self, x, full=False):
        """
        This function computes the fourth derivative of the basis function matrix for the points specified by x.

        Parameters
        ----------
        x : array-like
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        d4H : array-like
            Fourth derivative of the basis function matrix.
        """
        return np.hstack([k._Hjax(x, d=4, full=full) for j, k in enumerate(self._tfcClasses)])

    def d8H(self, x, full=False):
        """
        This function computes the eighth derivative of the basis function matrix for the points specified by x.

        Parameters
        ----------
        x : array-like
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        d8H : array-like
            Eighth derivative of the basis function matrix.
        """
        return np.hstack([k._Hjax(x, d=8, full=full) for j, k in enumerate(self._tfcClasses)])
