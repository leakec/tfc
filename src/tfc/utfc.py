import sys

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
            self.nC = nC
        elif isinstance(nC, list):
            self.nC = np.array(nC)
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
        return self._Hjax(x, full=full)

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
        return self._dHjax(x, full=full)

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
        return self._d2Hjax(x, full=full)

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
        return self._d4Hjax(x, full=full)

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
        return self._d8Hjax(x, full=full)

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
        dH_p = core.Primitive("dH")
        d2H_p = core.Primitive("d2H")
        d3H_p = core.Primitive("d3H")
        d4H_p = core.Primitive("d4H")
        d5H_p = core.Primitive("d5H")
        d6H_p = core.Primitive("d6H")
        d7H_p = core.Primitive("d7H")
        d8H_p = core.Primitive("d8H")

        def Hjax(x, full=False):
            return H_p.bind(x, full=full)

        def dHjax(x, full=False):
            return dH_p.bind(x, full=full)

        def d2Hjax(x, full=False):
            return d2H_p.bind(x, full=full)

        def d3Hjax(x, full=False):
            return d3H_p.bind(x, full=full)

        def d4Hjax(x, full=False):
            return d4H_p.bind(x, full=full)

        def d5Hjax(x, full=False):
            return d5H_p.bind(x, full=full)

        def d6Hjax(x, full=False):
            return d6H_p.bind(x, full=full)

        def d7Hjax(x, full=False):
            return d7H_p.bind(x, full=full)

        def d8Hjax(x, full=False):
            return d8H_p.bind(x, full=full)

        # Implicit translation
        def H_impl(x, full=False):
            return self.basisClass.H(x.flatten(), 0, full)

        def dH_impl(x, full=False):
            return self.basisClass.H(x.flatten(), 1, full)

        def d2H_impl(x, full=False):
            return self.basisClass.H(x.flatten(), 2, full)

        def d3H_impl(x, full=False):
            return self.basisClass.H(x.flatten(), 3, full)

        def d4H_impl(x, full=False):
            return self.basisClass.H(x.flatten(), 4, full)

        def d5H_impl(x, full=False):
            return self.basisClass.H(x.flatten(), 5, full)

        def d6H_impl(x, full=False):
            return self.basisClass.H(x.flatten(), 6, full)

        def d7H_impl(x, full=False):
            return self.basisClass.H(x.flatten(), 7, full)

        def d8H_impl(x, full=False):
            return self.basisClass.H(x.flatten(), 8, full)

        H_p.def_impl(H_impl)
        dH_p.def_impl(dH_impl)
        d2H_p.def_impl(d2H_impl)
        d3H_p.def_impl(d3H_impl)
        d4H_p.def_impl(d4H_impl)
        d5H_p.def_impl(d5H_impl)
        d6H_p.def_impl(d6H_impl)
        d7H_p.def_impl(d7H_impl)
        d8H_p.def_impl(d8H_impl)

        # Abstract evaluation
        def H_abstract_eval(x, full=False):
            dim0 = x.shape[0]
            if full:
                dim1 = self.basisClass.m
            else:
                dim1 = self.basisClass.m - self.basisClass.numC
            return abstract_arrays.ShapedArray((dim0, dim1), x.dtype)

        H_p.def_abstract_eval(H_abstract_eval)
        dH_p.def_abstract_eval(H_abstract_eval)
        d2H_p.def_abstract_eval(H_abstract_eval)
        d3H_p.def_abstract_eval(H_abstract_eval)
        d4H_p.def_abstract_eval(H_abstract_eval)
        d5H_p.def_abstract_eval(H_abstract_eval)
        d6H_p.def_abstract_eval(H_abstract_eval)
        d7H_p.def_abstract_eval(H_abstract_eval)
        d8H_p.def_abstract_eval(H_abstract_eval)

        # XLA compilation
        def H_xla(c, x, full=False):
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
                    _constant_s32_scalar(c, 0),
                    _constant_bool(c, full),
                    _constant_s32_scalar(c, dim0),
                    _constant_s32_scalar(c, dim1),
                ),
                xla_client.Shape.array_shape(dtype, (dim0, dim1)),
            )

        def dH_xla(c, x, full=False):
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
                    _constant_s32_scalar(c, 1),
                    _constant_bool(c, full),
                    _constant_s32_scalar(c, dim0),
                    _constant_s32_scalar(c, dim1),
                ),
                xla_client.Shape.array_shape(dtype, (dim0, dim1)),
            )

        def d2H_xla(c, x, full=False):
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
                    _constant_s32_scalar(c, 2),
                    _constant_bool(c, full),
                    _constant_s32_scalar(c, dim0),
                    _constant_s32_scalar(c, dim1),
                ),
                xla_client.Shape.array_shape(dtype, (dim0, dim1)),
            )

        def d3H_xla(c, x, full=False):
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
                    _constant_s32_scalar(c, 3),
                    _constant_bool(c, full),
                    _constant_s32_scalar(c, dim0),
                    _constant_s32_scalar(c, dim1),
                ),
                xla_client.Shape.array_shape(dtype, (dim0, dim1)),
            )

        def d4H_xla(c, x, full=False):
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
                    _constant_s32_scalar(c, 4),
                    _constant_bool(c, full),
                    _constant_s32_scalar(c, dim0),
                    _constant_s32_scalar(c, dim1),
                ),
                xla_client.Shape.array_shape(dtype, (dim0, dim1)),
            )

        def d5H_xla(c, x, full=False):
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
                    _constant_s32_scalar(c, 5),
                    _constant_bool(c, full),
                    _constant_s32_scalar(c, dim0),
                    _constant_s32_scalar(c, dim1),
                ),
                xla_client.Shape.array_shape(dtype, (dim0, dim1)),
            )

        def d6H_xla(c, x, full=False):
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
                    _constant_s32_scalar(c, 6),
                    _constant_bool(c, full),
                    _constant_s32_scalar(c, dim0),
                    _constant_s32_scalar(c, dim1),
                ),
                xla_client.Shape.array_shape(dtype, (dim0, dim1)),
            )

        def d7H_xla(c, x, full=False):
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
                    _constant_s32_scalar(c, 7),
                    _constant_bool(c, full),
                    _constant_s32_scalar(c, dim0),
                    _constant_s32_scalar(c, dim1),
                ),
                xla_client.Shape.array_shape(dtype, (dim0, dim1)),
            )

        def d8H_xla(c, x, full=False):
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
                    _constant_s32_scalar(c, 8),
                    _constant_bool(c, full),
                    _constant_s32_scalar(c, dim0),
                    _constant_s32_scalar(c, dim1),
                ),
                xla_client.Shape.array_shape(dtype, (dim0, dim1)),
            )

        xla.backend_specific_translations["cpu"][H_p] = H_xla
        xla.backend_specific_translations["cpu"][dH_p] = dH_xla
        xla.backend_specific_translations["cpu"][d2H_p] = d2H_xla
        xla.backend_specific_translations["cpu"][d3H_p] = d3H_xla
        xla.backend_specific_translations["cpu"][d4H_p] = d4H_xla
        xla.backend_specific_translations["cpu"][d5H_p] = d5H_xla
        xla.backend_specific_translations["cpu"][d6H_p] = d6H_xla
        xla.backend_specific_translations["cpu"][d7H_p] = d7H_xla
        xla.backend_specific_translations["cpu"][d8H_p] = d8H_xla

        # Define batching translation
        def H_batch(vec, batch, full=False):
            return Hjax(*vec, full=full), batch[0]

        def dH_batch(vec, batch, full=False):
            return dHjax(*vec, full=full), batch[0]

        def d2H_batch(vec, batch, full=False):
            return d2Hjax(*vec, full=full), batch[0]

        def d3H_batch(vec, batch, full=False):
            return d3Hjax(*vec, full=full), batch[0]

        def d4H_batch(vec, batch, full=False):
            return d4Hjax(*vec, full=full), batch[0]

        def d5H_batch(vec, batch, full=False):
            return d5Hjax(*vec, full=full), batch[0]

        def d6H_batch(vec, batch, full=False):
            return d6Hjax(*vec, full=full), batch[0]

        def d7H_batch(vec, batch, full=False):
            return d7Hjax(*vec, full=full), batch[0]

        def d8H_batch(vec, batch, full=False):
            return d8Hjax(*vec, full=full), batch[0]

        batching.primitive_batchers[H_p] = H_batch
        batching.primitive_batchers[dH_p] = dH_batch
        batching.primitive_batchers[d2H_p] = d2H_batch
        batching.primitive_batchers[d3H_p] = d3H_batch
        batching.primitive_batchers[d4H_p] = d4H_batch
        batching.primitive_batchers[d5H_p] = d5H_batch
        batching.primitive_batchers[d6H_p] = d6H_batch
        batching.primitive_batchers[d7H_p] = d7H_batch
        batching.primitive_batchers[d8H_p] = d8H_batch

        # Define jacobain vector product
        def H_jvp(arg_vals, arg_tans, full=False):
            x = arg_vals[0]
            dx = arg_tans[0]
            if not (dx is ad.Zero):
                if type(dx) is batching.BatchTracer:
                    flag = onp.any(dx.val != 0)
                else:
                    flag = onp.any(dx != 0)
                if flag:
                    if len(dx.shape) == 1:
                        out_tans = dHjax(x.flatten(), full=full) * np.expand_dims(dx, 1)
                    else:
                        out_tans = dHjax(x.flatten(), full=full) * dx
            else:
                dim0 = x.shape[0]
                if full:
                    dim1 = self.basisClass.m
                else:
                    dim1 = self.basisClass.m - self.basisClass.numC
                out_tans = np.zeros((dim0, dim1))
            return (Hjax(x.flatten(), full=full), out_tans)

        def dH_jvp(arg_vals, arg_tans, full=False):
            x = arg_vals[0]
            dx = arg_tans[0]
            if not (dx is ad.Zero):
                if type(dx) is batching.BatchTracer:
                    flag = onp.any(dx.val != 0)
                else:
                    flag = onp.any(dx != 0)
                if flag:
                    if len(dx.shape) == 1:
                        out_tans = d2Hjax(x.flatten(), full=full) * np.expand_dims(dx, 1)
                    else:
                        out_tans = d2Hjax(x.flatten(), full=full) * dx
            else:
                dim0 = x.shape[0]
                if full:
                    dim1 = self.basisClass.m
                else:
                    dim1 = self.basisClass.m - self.basisClass.numC
                out_tans = np.zeros((dim0, dim1))
            return (dHjax(x.flatten(), full=full), out_tans)

        def d2H_jvp(arg_vals, arg_tans, full=False):
            x = arg_vals[0]
            dx = arg_tans[0]
            if not (dx is ad.Zero):
                if type(dx) is batching.BatchTracer:
                    flag = onp.any(dx.val != 0)
                else:
                    flag = onp.any(dx != 0)
                if flag:
                    if len(dx.shape) == 1:
                        out_tans = d3Hjax(x.flatten(), full=full) * np.expand_dims(dx, 1)
                    else:
                        out_tans = d3Hjax(x.flatten(), full=full) * dx
            else:
                dim0 = x.shape[0]
                if full:
                    dim1 = self.basisClass.m
                else:
                    dim1 = self.basisClass.m - self.basisClass.numC
                out_tans = np.zeros((dim0, dim1))
            return (d2Hjax(x.flatten(), full=full), out_tans)

        def d3H_jvp(arg_vals, arg_tans, full=False):
            x = arg_vals[0]
            dx = arg_tans[0]
            if not (dx is ad.Zero):
                if type(dx) is batching.BatchTracer:
                    flag = onp.any(dx.val != 0)
                else:
                    flag = onp.any(dx != 0)
                if flag:
                    if len(dx.shape) == 1:
                        out_tans = d4Hjax(x.flatten(), full=full) * np.expand_dims(dx, 1)
                    else:
                        out_tans = d4Hjax(x.flatten(), full=full) * dx
            else:
                dim0 = x.shape[0]
                if full:
                    dim1 = self.basisClass.m
                else:
                    dim1 = self.basisClass.m - self.basisClass.numC
                out_tans = np.zeros((dim0, dim1))
            return (d3Hjax(x.flatten(), full=full), out_tans)

        def d4H_jvp(arg_vals, arg_tans, full=False):
            x = arg_vals[0]
            dx = arg_tans[0]
            if not (dx is ad.Zero):
                if type(dx) is batching.BatchTracer:
                    flag = onp.any(dx.val != 0)
                else:
                    flag = onp.any(dx != 0)
                if flag:
                    if len(dx.shape) == 1:
                        out_tans = d5Hjax(x.flatten(), full=full) * np.expand_dims(dx, 1)
                    else:
                        out_tans = d5Hjax(x.flatten(), full=full) * dx
            else:
                dim0 = x.shape[0]
                if full:
                    dim1 = self.basisClass.m
                else:
                    dim1 = self.basisClass.m - self.basisClass.numC
                out_tans = np.zeros((dim0, dim1))
            return (d4Hjax(x.flatten(), full=full), out_tans)

        def d5H_jvp(arg_vals, arg_tans, full=False):
            x = arg_vals[0]
            dx = arg_tans[0]
            if not (dx is ad.Zero):
                if type(dx) is batching.BatchTracer:
                    flag = onp.any(dx.val != 0)
                else:
                    flag = onp.any(dx != 0)
                if flag:
                    if len(dx.shape) == 1:
                        out_tans = d6Hjax(x.flatten(), full=full) * np.expand_dims(dx, 1)
                    else:
                        out_tans = d6Hjax(x.flatten(), full=full) * dx
            else:
                dim0 = x.shape[0]
                if full:
                    dim1 = self.basisClass.m
                else:
                    dim1 = self.basisClass.m - self.basisClass.numC
                out_tans = np.zeros((dim0, dim1))
            return (d5Hjax(x.flatten(), full=full), out_tans)

        def d6H_jvp(arg_vals, arg_tans, full=False):
            x = arg_vals[0]
            dx = arg_tans[0]
            if not (dx is ad.Zero):
                if type(dx) is batching.BatchTracer:
                    flag = onp.any(dx.val != 0)
                else:
                    flag = onp.any(dx != 0)
                if flag:
                    if len(dx.shape) == 1:
                        out_tans = d7Hjax(x.flatten(), full=full) * np.expand_dims(dx, 1)
                    else:
                        out_tans = d7Hjax(x.flatten(), full=full) * dx
            else:
                dim0 = x.shape[0]
                if full:
                    dim1 = self.basisClass.m
                else:
                    dim1 = self.basisClass.m - self.basisClass.numC
                out_tans = np.zeros((dim0, dim1))
            return (d6Hjax(x.flatten(), full=full), out_tans)

        def d7H_jvp(arg_vals, arg_tans, full=False):
            x = arg_vals[0]
            dx = arg_tans[0]
            if not (dx is ad.Zero):
                if type(dx) is batching.BatchTracer:
                    flag = onp.any(dx.val != 0)
                else:
                    flag = onp.any(dx != 0)
                if flag:
                    if len(dx.shape) == 1:
                        out_tans = d8Hjax(x.flatten(), full=full) * np.expand_dims(dx, 1)
                    else:
                        out_tans = d8Hjax(x.flatten(), full=full) * dx
            else:
                dim0 = x.shape[0]
                if full:
                    dim1 = self.basisClass.m
                else:
                    dim1 = self.basisClass.m - self.basisClass.numC
                out_tans = np.zeros((dim0, dim1))
            return (d7Hjax(x.flatten(), full=full), out_tans)

        ad.primitive_jvps[H_p] = H_jvp
        ad.primitive_jvps[dH_p] = dH_jvp
        ad.primitive_jvps[d2H_p] = d2H_jvp
        ad.primitive_jvps[d3H_p] = d3H_jvp
        ad.primitive_jvps[d4H_p] = d4H_jvp
        ad.primitive_jvps[d5H_p] = d5H_jvp
        ad.primitive_jvps[d6H_p] = d6H_jvp
        ad.primitive_jvps[d7H_p] = d7H_jvp

        # Provide pointers from TFC class
        self._Hjax = Hjax
        self._dHjax = dHjax
        self._d2Hjax = d2Hjax
        self._d3Hjax = d3Hjax
        self._d4Hjax = d4Hjax
        self._d8Hjax = d5Hjax


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
        return np.hstack([k._Hjax(x, full=full) for j, k in enumerate(self._tfcClasses)])

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
        return np.hstack([k._dHjax(x, full=full) for j, k in enumerate(self._tfcClasses)])

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
        return np.hstack([k._d2Hjax(x, full=full) for j, k in enumerate(self._tfcClasses)])

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
        return np.hstack([k._d3Hjax(x, full=full) for j, k in enumerate(self._tfcClasses)])

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
        return np.hstack([k._d4Hjax(x, full=full) for j, k in enumerate(self._tfcClasses)])

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
        return np.hstack([k._d8Hjax(x, full=full) for j, k in enumerate(self._tfcClasses)])
