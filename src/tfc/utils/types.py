import sys
from typing import Union, Any, Callable
import numpy as np
import numpy.typing as npt
from jax import Array
from sympy.core.function import AppliedUndef
from sympy import Expr

from typing import Literal, Protocol, TypedDict, Annotated

from annotated_types import Gt, Ge, Lt, Le

# Path
# Path = Union[str, os.PathLike]
Path = str

# Integer > 0
pint = Annotated[int, Gt(0)]

# Integer >= 0
uint = Annotated[int, Ge(0)]

# General number type
Number = Union[int, float, complex]

from numpy._typing._array_like import _ArrayLikeStr_co, _ArrayLikeInt_co

# Array-like of strings
StrArrayLike = _ArrayLikeStr_co

# Array-like of integers
IntArrayLike = _ArrayLikeInt_co

# List or array like
NumberListOrArray = Union[tuple[Number, ...], list[Number], npt.NDArray[Any], Array]

# List or array of integers
IntListOrArray = Union[
    tuple[int, ...],
    list[int],
    npt.NDArray[np.int32],
    npt.NDArray[np.int64],
    npt.NDArray[np.int16],
    npt.NDArray[np.int8],
]

# JAX array or numpy array
JaxOrNumpyArray = Union[npt.NDArray, Array]

# Tuple or list of array
TupleOrListOfArray = Union[tuple[JaxOrNumpyArray, ...], list[JaxOrNumpyArray]]
TupleOrListOfNumpyArray = Union[tuple[npt.NDArray, ...], list[npt.NDArray]]

# Sympy constraint operator
# Adding in Any here since sympy types are a bit funky at the moment
ConstraintOperator = Callable[[Union[AppliedUndef, Expr, Any]], Union[AppliedUndef, Any]]
ConstraintOperators = Union[list[ConstraintOperator], tuple[ConstraintOperator, ...]]

# List or tuple of sympy expressions
# Adding in Any here since sympy types are a bit funky at the moment
Exprs = Union[list[Union[Expr, Any]], tuple[Union[Expr, Any], ...]]
