import sys
from typing import Union, Any, Callable
import numpy as np
import numpy.typing as npt
import jax.numpy as jnp
from jaxtyping import Array
from sympy.core.function import AppliedUndef
from sympy import Expr

if sys.version_info >= (3, 8):
    from typing import Literal, Protocol, TypedDict
else:
    from typing_extensions import Literal, Protocol, TypedDict

if sys.version_info >= (3, 9):
    from typing import Annotated

    List = list
    Tuple = tuple
    Dict = dict
    Tuple = tuple
else:
    from typing_extensions import Annotated
    from typing import List, Tuple, Dict, Tuple

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

if sys.version_info >= (3, 8):
    from numpy._typing._array_like import _ArrayLikeStr_co, _ArrayLikeInt_co

    # Array-like of strings
    StrArrayLike = _ArrayLikeStr_co

    # Array-like of integers
    IntArrayLike = _ArrayLikeInt_co
else:
    # Hacks to keep things working for Python 3.7
    StrArrayLike = Any
    IntArrayLike = Any

# List or array like
NumberListOrArray = Union[Tuple[Number, ...], List[Number], npt.NDArray[Any], Array]

# List or array of integers
IntListOrArray = Union[
    Tuple[int, ...],
    List[int],
    npt.NDArray[np.int32],
    npt.NDArray[np.int64],
    npt.NDArray[np.int16],
    npt.NDArray[np.int8],
]

# JAX array or numpy array
JaxOrNumpyArray = Union[npt.NDArray, Array]

# Tuple or list of array
TupleOrListOfArray = Union[Tuple[JaxOrNumpyArray, ...], List[JaxOrNumpyArray]]
TupleOrListOfNumpyArray = Union[Tuple[npt.NDArray, ...], List[npt.NDArray]]

# Sympy constraint operator
# Adding in Any here since sympy types are a bit funky at the moment
ConstraintOperator = Callable[[Union[AppliedUndef, Expr, Any]], Union[AppliedUndef, Any]]
ConstraintOperators = Union[List[ConstraintOperator], Tuple[ConstraintOperator, ...]]

# List or tuple of sympy expressions
# Adding in Any here since sympy types are a bit funky at the moment
Exprs = Union[List[Union[Expr, Any]], Tuple[Union[Expr, Any], ...]]
