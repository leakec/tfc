from jax._src.config import config

config.update("jax_enable_x64", True)

import numpy as onp
import jax.numpy as np
import numpy.typing as npt
from typing import Optional, cast
from .utils.tfc_types import Literal, uint, IntArrayLike, JaxOrNumpyArray
from jax import core
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir
from jax.ffi import register_ffi_target
import jaxlib.mlir.ir as ir

# This is not part of the public API. However, it is what JAX uses internally in the ffi
# interface. We need this here, since we want to do very low-level things, like injecting
# new operands that are not traced into the C++ code.
# To switch to the new FFI interface, we would need to re-work all the C++ code to take
# in arguments as a JSON string. This would make the C++ way more confusing than it needs to be.
from jax._src.interpreters import mlir as mlir_int

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
    nC : IntArrayLike
        Number of functions to remove from the free function linear expansion. This variable is used to account for basis functions that are linearly dependent on support functions used in the construction of the constrained expressions. It can be expressed in 1 of 2 ways.

        1. As an integer. When expressed as an integer, the first nC basis functions are removed from the free function.
        2. As a list or array. When expressed as a list or array, the basis functions corresponding to the numbers given by the list or array are removed from the free function.

    deg : int
        Degree of the basis function expansion. This number is one less than the number of basis functions used before removing those specified by nC.
    x0 : float, optional
        Specifies the beginning of the DE domain. (Default value = 0)
    xf : float
        This required keyword argument specifies the end of the DE domain.
    basis : Literal["CP","LeP","FS","ELMTanh","ELMSigmoid","ELMSin","ELMSwish","ELMReLU"], optional
        This optional keyword argument specifies the basis functions to be used. (Default value = "CP")
    """

    def __init__(
        self,
        N: uint,
        nC: IntArrayLike,
        deg: uint,
        basis: Literal[
            "CP", "LeP", "FS", "ELMTanh", "ELMSigmoid", "ELMSin", "ELMSwish", "ELMReLU"
        ] = "CP",
        x0: Optional[float] = None,
        xf: Optional[float] = None,
        backend: Literal["C++", "Python"] = "C++",
    ):
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
        deg : int
            Degree of the basis function expansion. This number is one less than the number of basis functions used before removing those specified by nC.
        x0 : float, optional
            Specifies the beginning of the DE domain. (Default value = 0)
        xf : float
            This required keyword argument specifies the end of the DE domain.
        basis : {"CP","LeP","FS","ELMTanh","ELMSigmoid","ELMSin","ELMSwish","ELMReLU"}, optional
            This optional keyword argument specifies the basis functions to be used. (Default value = "CP")
        backend : Literal["C++", "Python"]
            This optional keyword sets the backend used to compute the basis functions. The C++ can be used with JIT, but can only be used for doubles. The Python backend can be used for other field types, e.g., complex numbers, but does not have JIT translations. Instead, pejit must be used to set the basis function outputs as compile time constants in order to JIT.
        """

        # Store givens
        self.N = N
        self.deg = deg
        self._backend = backend

        if isinstance(nC, int):
            self.nC: npt.NDArray = onp.arange(nC, dtype=onp.int32)
        elif isinstance(nC, np.ndarray):
            self.nC: npt.NDArray = cast(npt.NDArray, nC.astype(onp.int32))
        elif isinstance(nC, list):
            self.nC: npt.NDArray = np.array(nC, dtype=np.int32)
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

        if isinstance(self.x0, int):
            self.x0 = float(self.x0)
            TFCPrint.Warning("x0 is an integer. Converting to float to avoid errors down the line.")

        if xf is None:
            self.xf = 0.0
        else:
            self.xf = xf

        if isinstance(self.xf, int):
            self.xf = float(self.xf)
            TFCPrint.Warning("xf is an integer. Converting to float to avoid errors down the line.")

        # Setup the basis function
        if backend == "C++":
            from .utils import BF
        elif backend == "Python":
            from .utils import BF_Py as BF
        else:
            TFCPrint.Error(
                f'The backend {backend} was specified, but can only be one of "C++" or "Python".'
            )
        if self.basis == "CP":
            self.basisClass = BF.CP(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = -1.0
            zf = 1.0
        elif self.basis == "LeP":
            self.basisClass = BF.LeP(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = -1.0
            zf = 1.0
        elif self.basis == "FS":
            self.basisClass = BF.FS(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = -np.pi
            zf = np.pi
        elif self.basis == "ELMReLU":
            self.basisClass = BF.ELMReLU(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = 0.0
            zf = 1.0
        elif self.basis == "ELMSigmoid":
            self.basisClass = BF.ELMSigmoid(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = 0.0
            zf = 1.0
        elif self.basis == "ELMTanh":
            self.basisClass = BF.ELMTanh(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = 0.0
            zf = 1.0
        elif self.basis == "ELMSin":
            self.basisClass = BF.ELMSin(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = 0.0
            zf = 1.0
        elif self.basis == "ELMSwish":
            self.basisClass = BF.ELMSwish(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = 0.0
            zf = 1.0
        else:
            TFCPrint.Error("Invalid basis selection. Please select a valid basis")

        self.c = self.basisClass.c

        # Calculate z points and corresponding x
        if self.basis in ["CP", "LeP"]:
            n = self.N - 1
            # Multiplying x0 by 0 below so the array I has the same
            # type as x0.
            I = np.linspace(0 * self.x0, n, n + 1)
            self.z = np.cos(np.pi * (n - I) / float(n))
            self.x = (self.z - z0) / self.c + self.x0
        else:
            self.z = np.linspace(z0, zf, self.N)
            self.x = (self.z - z0) / self.c + self.x0

        self._SetupJax()

    def H(self, x: JaxOrNumpyArray, full: bool = False) -> npt.NDArray:
        """
        This function computes the basis function matrix for the points specified by x.

        Parameters
        ----------
        x : JaxOrNumpyArray
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        H : NDArray
            Basis function matrix.
        """
        return self._Hjax(x, d=0, full=full)

    def dH(self, x: JaxOrNumpyArray, full: bool = False) -> npt.NDArray:
        """This function computes the deriative of H. See documentation of 'H' for more details.

        Parameters
        ----------
        x : JaxOrNumpyArray
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        H : NDArray
            Derivative of the basis function matrix.
        """
        return self._Hjax(x, d=1, full=full)

    def d2H(self, x: JaxOrNumpyArray, full: bool = False) -> npt.NDArray:
        """This function computes the second deriative of H. See documentation of H for more details.

        Parameters
        ----------
        x : JaxOrNumpyArray
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        d2H : NDArray
            Second derivative of the basis function matrix.
        """
        return self._Hjax(x, d=2, full=full)

    def d4H(self, x: JaxOrNumpyArray, full: bool = False) -> npt.NDArray:
        """This function computes the fourth deriative of H. See documentation of H for more details.

        Parameters
        ----------
        x : JaxOrNumpyArray
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        d4H : NDArray
            Fourth derivative of the basis function matrix.
        """
        return self._Hjax(x, d=4, full=full)

    def d8H(self, x: JaxOrNumpyArray, full: bool = False) -> npt.NDArray:
        """This function computes the eighth deriative of H. See documentation of H for more details.

        Parameters
        ----------
        x : JaxOrNumpyArray
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        d8H : NDArray
            Eighth derivative of the basis function matrix.
        """
        return self._Hjax(x, d=8, full=full)

    def _SetupJax(self):
        """This function is used internally by TFC to setup JAX primatives and create desired behavior when taking derivatives of TFC constrained expressions."""

        # Regiser XLA function
        if self._backend == "C++":
            obj = self.basisClass.xlaCapsule
            xlaName = "BasisFunc" + str(self.basisClass.identifier)
            register_ffi_target(xlaName, obj, platform="cpu", api_version=0)

        # Create primitives
        H_p = Primitive("H")

        def Hjax(x: JaxOrNumpyArray, d: uint = 0, full: bool = False) -> npt.NDArray:
            return cast(npt.NDArray, H_p.bind(x, d=d, full=full))

        # Implicit translation
        def H_impl(x: npt.NDArray, d: uint = 0, full=False) -> npt.NDArray:
            return self.basisClass.H(x, d, full)

        H_p.def_impl(H_impl)

        # Abstract evaluation
        def H_abstract_eval(x, d: uint = 0, full: bool = False) -> core.ShapedArray:
            if full:
                dim1 = self.basisClass.m
            else:
                dim1 = self.basisClass.m - self.basisClass.numC
            if len(x.shape) == 0:
                dims = (dim1,)
            else:
                dims = (x.shape[0], dim1)
            return core.ShapedArray(dims, x.dtype)

        H_p.def_abstract_eval(H_abstract_eval)

        if self._backend == "C++":
            # XLA compilation

            def H_xla(ctx, x, d: uint = 0, full: bool = False):
                x_ir_type = ir.RankedTensorType(x.type)  # x.type is already an ir.Type
                x_element_type = x_ir_type.element_type
                x_dims = x_ir_type.shape  # This is a list of integers

                dim0 = x_dims[0]
                if full:
                    dim1 = self.basisClass.m
                else:
                    dim1 = self.basisClass.m - self.basisClass.numC

                # Define Result Types
                result_types = [ir.RankedTensorType.get([dim0, dim1], x_element_type)]

                # Call mlir.custom_call
                custom_call_op = mlir_int.custom_call(
                    call_target_name=xlaName,
                    result_types=result_types,
                    operands=[
                        mlir.ir_constant(np.int32(self.basisClass.identifier)),
                        x,
                        mlir.ir_constant(np.int32(d)),
                        mlir.ir_constant(bool(full)),
                        mlir.ir_constant(np.int32(dim0)),
                        mlir.ir_constant(np.int32(dim1)),
                    ],
                    has_side_effect=False,
                    api_version=3,
                )

                return custom_call_op.results

            mlir.register_lowering(H_p, H_xla, platform="cpu")

        # Define batching translation
        def H_batch(vec, batch, d: uint = 0, full: bool = False):
            return Hjax(*vec, d=d, full=full), batch[0]

        batching.primitive_batchers[H_p] = H_batch

        # Define jacobain vector product
        def H_jvp(arg_vals, arg_tans, d: uint = 0, full: bool = False):
            x = arg_vals[0]
            dx = arg_tans[0]
            if not (dx is ad.Zero):
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

    def H(self, x: JaxOrNumpyArray, full: bool = False) -> npt.NDArray:
        """
        This function computes the basis function matrix for the points specified by x.

        Parameters
        ----------
        x : JaxOrNumpyArray
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        H : NDArray
            Basis function matrix.
        """
        return cast(
            npt.NDArray,
            np.hstack([k._Hjax(x, d=0, full=full) for j, k in enumerate(self._tfcClasses)]),
        )

    def dH(self, x: JaxOrNumpyArray, full: bool = False) -> npt.NDArray:
        """
        This function computes the derivative of the basis function matrix for the points specified by x.

        Parameters
        ----------
        x : JaxOrNumpyArray
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        dH : NDArray
            Derivative of the basis function matrix.
        """
        return cast(
            npt.NDArray,
            np.hstack([k._Hjax(x, d=1, full=full) for j, k in enumerate(self._tfcClasses)]),
        )

    def d2H(self, x: JaxOrNumpyArray, full: bool = False) -> npt.NDArray:
        """
        This function computes the second derivative of the basis function matrix for the points specified by x.

        Parameters
        ----------
        x : JaxOrNumpyArray
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        d2H : NDArray
            Second derivative of the basis function matrix.
        """
        return cast(
            npt.NDArray,
            np.hstack([k._Hjax(x, d=2, full=full) for j, k in enumerate(self._tfcClasses)]),
        )

    def d3H(self, x: JaxOrNumpyArray, full: bool = False) -> npt.NDArray:
        """
        This function computes the third derivative of the basis function matrix for the points specified by x.

        Parameters
        ----------
        x : JaxOrNumpyArray
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        d3H : NDArray
            Third derivative of the basis function matrix.
        """
        return cast(
            npt.NDArray,
            np.hstack([k._Hjax(x, d=3, full=full) for j, k in enumerate(self._tfcClasses)]),
        )

    def d4H(self, x: JaxOrNumpyArray, full: bool = False) -> npt.NDArray:
        """
        This function computes the fourth derivative of the basis function matrix for the points specified by x.

        Parameters
        ----------
        x : JaxOrNumpyArray
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        d4H : NDArray
            Fourth derivative of the basis function matrix.
        """
        return cast(
            npt.NDArray,
            np.hstack([k._Hjax(x, d=4, full=full) for j, k in enumerate(self._tfcClasses)]),
        )

    def d8H(self, x: JaxOrNumpyArray, full: bool = False) -> npt.NDArray:
        """
        This function computes the eighth derivative of the basis function matrix for the points specified by x.

        Parameters
        ----------
        x : JaxOrNumpyArray
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        d8H : NDArray
            Eighth derivative of the basis function matrix.
        """
        return cast(
            npt.NDArray,
            np.hstack([k._Hjax(x, d=8, full=full) for j, k in enumerate(self._tfcClasses)]),
        )
