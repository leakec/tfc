from jax._src.config import config

config.update("jax_enable_x64", True)

from copy import copy
import numpy as onp
import jax.numpy as np
from typing import cast
import numpy.typing as npt
from .utils.types import (
    Literal,
    uint,
    IntListOrArray,
    pint,
    NumberListOrArray,
    JaxOrNumpyArray,
    IntArrayLike,
    Array,
    Tuple,
)
from jax import core
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir
from jax.ffi import register_ffi_target
import jaxlib.mlir.ir as ir
from jaxlib.mlir.dialects import stablehlo

# This is not part of the public API. However, it is what JAX uses internally in the ffi
# interface. We need this here, since we want to do very low-level things, like injecting
# new operands that are not traced into the C++ code.
# To switch to the new FFI interface, we would need to re-work all the C++ code to take
# in arguments as a JSON string. This would make the C++ way more confusing than it needs to be.
from jax._src.interpreters import mlir as mlir_int

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
    n : IntListOrArray
        Number of points to use per-dimension when discretizing the domain. List or array must be same length as number of dimensions.

    nC : IntArrayLike
        Number of functions to remove from the beginning of free function linear expansion. This variable is used to account for basis functions that are linearly dependent on support functions used in the construction of the constrained expressions. The constraints for each dimension can be expressed in 1 of 2 ways. Note that a value of -1 is used to indicate no constraints exist for a particular dimension.

        1. As an integer. When expressed as an integer, the first nC basis functions are removed from the free function.
        2. As a list or array. When expressed as a list or array, the basis functions corresponding to the numbers given by the list or array are removed from the free function.

    deg : uint
        Degree of the basis function expansion.

    dim : pint
        Number of dimensions in the domain.

    basis : Literal["CP","LeP","FS","ELMTanh","ELMSigmoid","ELMSin","ELMSwish","ELMReLU"], optional
        This optional keyword argument specifies the basis functions to be used. (Default value = "CP")

    x0 : NumberListOrArray
        Specifies the beginning of the DE domain. (Default value = None)

    xf : NumberListOrArray
        Specifies the end of the DE domain. (Default value = None)

    backend : Literal["C++", "Python"]
        This optional keyword sets the backend used to compute the basis functions. The C++ can be used with JIT, but can only be used for doubles. The Python backend can be used for other field types, e.g., complex numbers, but does not have JIT translations. Instead, pejit must be used to set the basis function outputs as compile time constants in order to JIT.
    """

    def __init__(
        self,
        n: IntListOrArray,
        nC: IntArrayLike,
        deg: uint,
        dim: pint = 2,
        basis: Literal[
            "CP", "LeP", "FS", "ELMTanh", "ELMSigmoid", "ELMSin", "ELMSwish", "ELMReLU"
        ] = "CP",
        x0: NumberListOrArray = [],
        xf: NumberListOrArray = [],
        backend: Literal["C++", "Python"] = "C++",
    ):
        # Store givens
        self._elm_classes = ["ELMSigmoid", "ELMTanh", "ELMSin", "ELMSwish", "ELMReLU"]
        self.deg = deg
        self.dim = dim
        self._backend = backend

        _int_types = [onp.intp, onp.int8, onp.int16, onp.int32, onp.int64]

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
            self.x0 = onp.zeros(dim)
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
                self.x0 = cast(onp.ndarray, x0)
            else:
                if not len(x0) == dim:
                    TFCPrint.Error(
                        "x0 has length "
                        + str(len(x0))
                        + ", but it should be equal to the number of dimensions, "
                        + str(dim)
                        + "."
                    )
                self.x0 = onp.array(x0).flatten()
                if not self.x0.shape[0] == dim:
                    TFCPrint.Error(
                        "x0 has length "
                        + str(self.x0.shape[0])
                        + ", but it should be equal to the number of dimensions, "
                        + str(dim)
                        + "."
                    )

        if self.x0.dtype in _int_types:
            self.x0 = onp.array(self.x0, dtype=onp.float64)
            TFCPrint.Warning(
                "x0 is an integer type. Converting to float64 to avoid errors down the line."
            )

        # Set xf based on user input
        if xf is None:
            self.xf = onp.zeros(dim)
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
                self.xf = cast(onp.ndarray, xf)
            else:
                if not len(xf) == dim:
                    TFCPrint.Error(
                        "xf has length "
                        + str(len(xf))
                        + ", but it should be equal to the number of dimensions, "
                        + str(dim)
                        + "."
                    )
                self.xf = onp.array(xf).flatten()
                if not self.xf.shape[0] == dim:
                    TFCPrint.Error(
                        "xf has length "
                        + str(self.xf.shape[0])
                        + ", but it should be equal to the number of dimensions, "
                        + str(dim)
                        + "."
                    )
        if self.xf.dtype in _int_types:
            self.xf = onp.array(self.xf, dtype=onp.float64)
            TFCPrint.Warning(
                "xf is an integer type. Converting to float64 to avoid errors down the line."
            )

        # Create nC matrix based on user input
        if basis in self._elm_classes:
            if isinstance(nC, int):
                self.nC = onp.arange(nC, dtype=onp.int32)
            elif isinstance(nC, np.ndarray):
                self.nC = cast(onp.ndarray, nC.astype(onp.int32))
            elif isinstance(nC, list):
                self.nC = onp.array(nC, dtype=np.int32)
            if self.nC.shape[0] > self.deg:
                TFCPrint.Error("Number of basis functions is less than number of constraints!")
            if np.any(self.nC < 0):
                TFCPrint.Error(
                    "To set nC to -1 (no constraints) either use nC = -1 or nC = 0 (i.e., use an integer not a list or array). Do not put only -1 in a list or array, this will cause issues in the C++ layer."
                )
        else:
            if isinstance(nC, int):
                TFCPrint.Error(
                    "Cannot use type int for nC when specifying non-ELM type basis function."
                )
            # Using explicit type casts here to keep LSPs happy. At this point, we know nC is not a regular integer.
            if isinstance(nC, np.ndarray) and len(nC.shape) > 1:
                if not nC.shape[0] == self.dim:
                    TFCPrint.Error(
                        "nC has "
                        + str(nC.flatten().shape[0])
                        + " rows, but the row number should be equal to the number of dimensions, "
                        + str(dim)
                        + "."
                    )
                self.nC = cast(onp.ndarray, nC.astype(np.int32))
            else:
                if isinstance(nC, np.ndarray):
                    nC = nC.tolist()
                if not len(cast(IntListOrArray, nC)) == dim:
                    TFCPrint.Error(
                        "nC has length "
                        + str(len(cast(IntListOrArray, nC)))
                        + ", but it should be equal to the number of dimensions, "
                        + str(dim)
                        + "."
                    )
                nCmax = 0
                for k in range(dim):
                    if isinstance(cast(IntListOrArray, nC)[k], np.ndarray):
                        nCk = np.array(cast(IntListOrArray, nC)[k]).flatten()
                    else:
                        nCk = np.array([cast(IntListOrArray, nC)[k]]).flatten()
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
                    if isinstance(cast(IntListOrArray, nC)[k], np.ndarray):
                        nCk = np.array(cast(IntListOrArray, nC)[k]).flatten()
                    else:
                        nCk = onp.array([cast(IntListOrArray, nC)[k]]).flatten()
                    j = nCk.shape[0]
                    if j == 1:
                        nCk = onp.arange(nCk[0])
                        j = nCk.shape[0]
                    if j < nCmax:
                        if j == 0:
                            nCk = -1.0 * onp.ones(nCmax)
                        else:
                            nCk = cast(npt.NDArray, np.hstack([nCk, -1 * np.ones(nCmax - j)]))
                    onC[k, :] = nCk.astype(np.int32)
                self.nC = onp.array(onC.tolist(), dtype=np.int32)

        # Setup the basis function
        if backend == "C++":
            from .utils import BF
        elif backend == "Python":
            from .utils.BF import BF_Py as BF
        else:
            TFCPrint.Error(
                f'The backend {backend} was specified, but can only be one of "C++" or "Python".'
            )
        if self.basis == "CP":
            self.basisClass = BF.nCP(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = -1.0
            zf = 1.0
        elif self.basis == "LeP":
            self.basisClass = BF.nLeP(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = -1.0
            zf = 1.0
        elif self.basis == "FS":
            self.basisClass = BF.nFS(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = -np.pi
            zf = np.pi
        elif self.basis == "ELMSigmoid":
            self.basisClass = BF.nELMSigmoid(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = 0.0
            zf = 1.0
        elif self.basis == "ELMTanh":
            self.basisClass = BF.nELMTanh(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = 0.0
            zf = 1.0
        elif self.basis == "ELMSin":
            self.basisClass = BF.nELMSin(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = 0.0
            zf = 1.0
        elif self.basis == "ELMSwish":
            self.basisClass = BF.nELMSwish(self.x0, self.xf, self.nC, self.deg + 1)
            z0 = 0.0
            zf = 1.0
        elif self.basis == "ELMReLU":
            self.basisClass = BF.nELMReLU(self.x0, self.xf, self.nC, self.deg + 1)
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
        self.z = onp.zeros((self.dim, self.N), dtype=self.x0.dtype)
        x = tuple([onp.zeros(self.N, dtype=self.x0.dtype) for _ in range(self.dim)])
        if self.basis in ["CP", "LeP"]:
            for k in range(self.dim):
                nProd = int(onp.prod(self.n[k + 1 :]))
                nStack = int(onp.prod(self.n[0:k]))
                j = self.n[k] - 1
                # Multiplying x0 by 0 here so the array will have the
                # same type as x0.
                I = onp.linspace(0 * x0[0], j, j + 1).reshape((j + 1, 1))
                dark = onp.cos(np.pi * (j - I) / float(j))
                dark = onp.hstack([dark] * nProd).flatten()
                self.z[k, :] = onp.array([dark] * nStack).flatten()
                x[k][:] = (self.z[k, :] - z0) / self.c[k] + self.x0[k]
        else:
            for k in range(self.dim):
                nProd = int(onp.prod(self.n[k + 1 :]))
                nStack = int(onp.prod(self.n[0:k]))
                dark = onp.linspace(z0, zf, num=self.n[k], dtype=self.x0.dtype).reshape(
                    (self.n[k], 1)
                )
                dark = onp.hstack([dark] * nProd).flatten()
                self.z[k, :] = onp.array([dark] * nStack).flatten()
                x[k][:] = (self.z[k, :] - z0) / self.c[k] + self.x0[k]

        self.z: Array = cast(Array, np.array(self.z.tolist()))
        self.x: Tuple[Array, ...] = tuple(
            [cast(Array, np.array(x[k].tolist())) for k in range(self.dim)]
        )

        self.SetupJAX()

    def H(self, *x: JaxOrNumpyArray, full: bool = False) -> npt.NDArray:
        """
        This function computes the basis function matrix for the points specified by *x.

        Parameters
        ----------
        *x : JaxOrNumpyArray
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        H : NDArray
            Basis function matrix.
        """
        d = tuple(0 for _ in range(self.dim))
        return self._Hjax(*x, d=d, full=full)

    def Hx(self, *x: JaxOrNumpyArray, full: bool = False) -> npt.NDArray:
        """
        This function computes the derivative of the basis function matrix for the points specified by *x with respect to the first variable.

        Parameters
        ----------
        *x : JaxOrNumpyArray
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        Hx : NDArray
            Derivative of the basis function matrix with respect to the first variable.
        """
        d = tuple(1 if k == 0 else 0 for k in range(self.dim))
        return self._Hjax(*x, d=d, full=full)

    def Hx2(self, *x: JaxOrNumpyArray, full: bool = False) -> npt.NDArray:
        """
        This function computes the second derivative of the basis function matrix for the points specified by *x with respect to the first variable.

        Parameters
        ----------
        *x : JaxOrNumpyArray
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        Hx2 : NDArray
            Second derivative of the basis function matrix with respect to the first variable.
        """
        d = tuple(2 if k == 0 else 0 for k in range(self.dim))
        return self._Hjax(*x, d=d, full=full)

    def Hy2(self, *x: JaxOrNumpyArray, full: bool = False) -> npt.NDArray:
        """
        This function computes the second derivative of the basis function matrix for the points specified by *x with respect to the second variable.

        Parameters
        ----------
        *x : JaxOrNumpyArray
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        Hy2 : NDArray
            Second derivative of the basis function matrix with respect to the second variable.
        """
        d = tuple(2 if k == 1 else 0 for k in range(self.dim))
        return self._Hjax(*x, d=d, full=full)

    def Hx2y(self, *x: JaxOrNumpyArray, full: bool = False) -> npt.NDArray:
        """
        This function computes the mixed derivative (second order derivative with respect to the first variable and first order with respect
        to the second variable) of the basis function matrix for the points specified by *x.

        Parameters
        ----------
        *x : JaxOrNumpyArray
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        Hx2y : NDArray
            Mixed derivative of the basis function matrix with respect to the first variable.
        """
        d = [0 for _ in range(self.dim)]
        d[0] = 2
        d[1] = 1
        return self._Hjax(*x, d=tuple(d), full=full)

    def Hy(self, *x: JaxOrNumpyArray, full: bool = False) -> npt.NDArray:
        """
        This function computes the derivative of the basis function matrix for the points specified by *x with respect to the second variable.

        Parameters
        ----------
        *x : JaxOrNumpyArray
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        Hy : NDArray
            Derivative of the basis function matrix with respect to the second variable.
        """
        d = tuple(1 if k == 1 else 0 for k in range(self.dim))
        return self._Hjax(*x, d=d, full=full)

    def Hxy(self, *x: JaxOrNumpyArray, full: bool = False) -> npt.NDArray:
        """
        This function computes the mixed derivative (first order derivative with respect to the first variable and first order with respect
        to the second variable) of the basis function matrix for the points specified by *x.

        Parameters
        ----------
        *x : JaxOrNumpyArray
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        Hxy : NDArray
            Mixed derivative of the basis function matrix with respect to the first variable.
        """
        d = [0 for _ in range(self.dim)]
        d[0] = 1
        d[1] = 1
        return self._Hjax(*x, d=tuple(d), full=full)

    def Hz(self, *x: JaxOrNumpyArray, full: bool = False) -> npt.NDArray:
        """
        This function computes the derivative of the basis function matrix for the points specified by *x with respect to the third variable.

        Parameters
        ----------
        *x : JaxOrNumpyArray
            Points to calculate the basis functions at.

        full : bool, optional
            If true then the values specified by nC to the utfc class are ignored and all basis functions are computed. (Default value = False)

        Returns
        -------
        Hz : NDArray
            Derivative of the basis function matrix with respect to the third variable.
        """
        d = tuple(1 if k == 2 else 0 for k in range(self.dim))
        return self._Hjax(*x, d=d, full=full)

    def SetupJAX(self):
        """This function is used internally by TFC to setup autograd primatives and create desired behavior when taking derivatives of TFC constrained expressions."""

        # Helper variables
        d0 = tuple(0 for _ in range(self.dim))

        # Regiser XLA function
        if self._backend == "C++":
            obj = self.basisClass.xlaCapsule
            xlaName = "BasisFunc" + str(self.basisClass.identifier)
            register_ffi_target(xlaName, obj, platform="cpu", api_version=0)

        # Create Primitives
        H_p = Primitive("H")

        def Hjax(*x: JaxOrNumpyArray, d: tuple[int, ...] = d0, full: bool = False):
            return cast(npt.NDArray, H_p.bind(*x, d=d, full=full))

        # Implicit translations
        def H_impl(*x: npt.NDArray, d: tuple[int, ...] = d0, full: bool = False):
            return self.basisClass.H(np.array(x), d, full)

        H_p.def_impl(H_impl)

        # Define abstract evaluation
        def H_abstract_eval(
            *x, d: tuple[int, ...] = d0, full: bool = False
        ) -> core.ShapedArray:
            if full:
                dim1 = self.basisClass.numBasisFuncFull
            else:
                dim1 = self.basisClass.numBasisFunc
            if len(x[0].shape) == 0:
                dims = (dim1,)
            else:
                dims = (x[0].shape[0], dim1)
            return core.ShapedArray(dims, x[0].dtype)

        H_p.def_abstract_eval(H_abstract_eval)

        if self._backend == "C++":
            # XLA compilation
            def default_layout(shape):
                return tuple(range(len(shape) - 1, -1, -1))

            def H_xla(ctx, *x, d: uint = 0, full: bool = False):
                x_type = ir.RankedTensorType(x[0].type)
                dims = x_type.shape
                dim0 = dims[0]
                if full:
                    dim1 = self.basisClass.numBasisFuncFull
                else:
                    dim1 = self.basisClass.numBasisFunc
                res_types = [ir.RankedTensorType.get((dim0, dim1), x_type.element_type)]
                return mlir_int.custom_call(
                    call_target_name=xlaName,
                    result_types=res_types,
                    operands=[
                        mlir.ir_constant(np.int32(self.basisClass.identifier)),
                        stablehlo.ConcatenateOp(x, 0).result,
                        mlir.ir_constant(np.int32(d)),
                        mlir.ir_constant(np.int32(self.dim)),
                        mlir.ir_constant(bool(full)),
                        mlir.ir_constant(np.int32(dim0)),
                        mlir.ir_constant(np.int32(dim1)),
                    ],
                    operand_layouts=[
                        (),
                        default_layout((dim0 * len(x),)),
                        default_layout((len(d),)),
                        (),
                        (),
                        (),
                        (),
                    ],
                    result_layouts=[
                        default_layout((dim0, dim1)),
                    ],
                    api_version=3,
                ).results

            mlir.register_lowering(H_p, H_xla, platform="cpu")

        # Batching translation
        def H_batch(vec, batch, d: tuple[int, ...] = d0, full: bool = False):
            return Hjax(*vec, d=d, full=full), batch[0]

        batching.primitive_batchers[H_p] = H_batch

        # Jacobian vector translation
        def H_jvp(arg_vals, arg_tans, d: tuple[int, ...] = d0, full: bool = False):
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
                        dark = tuple(d[j]+1 if k == j else d[j] for j in range(len(d)))
                        if flat:
                            out_tans += Hjax(*arg_vals, d=dark, full=full) * np.expand_dims(
                                arg_tans[k], 1
                            )
                        else:
                            out_tans += Hjax(*arg_vals, d=dark, full=full) * arg_tans[k]
            return (Hjax(*arg_vals, d=d, full=full), out_tans)

        ad.primitive_jvps[H_p] = H_jvp

        self._Hjax = Hjax
