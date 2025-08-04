import sys
from typing import Any, Callable
import numpy as np
import numpy.typing as npt
from jax import Array
from sympy.core.function import AppliedUndef
from sympy import Expr

from typing import Literal, Protocol, TypedDict, Annotated

from annotated_types import Gt, Ge, Lt, Le

# Path
# Path = str | os.PathLike
Path = str

# Integer > 0
pint = Annotated[int, Gt(0)]

# Integer >= 0
uint = Annotated[int, Ge(0)]

# General number type
Number = int | float | complex

from numpy._typing._array_like import _ArrayLikeStr_co

# Array-like of strings
StrArrayLike = _ArrayLikeStr_co

# Array-like of integers
IntArrayLike = Annotated[npt.ArrayLike, np.int32]

# List or array like
NumberListOrArray = tuple[Number, ...] | list[Number] | npt.NDArray[Any] | Array

# List or array of integers
IntListOrArray = IntArrayLike

# JAX array or numpy array
JaxOrNumpyArray = npt.NDArray | Array

# Tuple or list of array
TupleOrListOfArray = tuple[JaxOrNumpyArray, ...] | list[JaxOrNumpyArray]
TupleOrListOfNumpyArray = tuple[npt.NDArray, ...] | list[npt.NDArray]

# Sympy constraint operator
# Adding in Any here since sympy types are a bit funky at the moment
ConstraintOperator = Callable[[AppliedUndef | Expr | Any], AppliedUndef | Any]
ConstraintOperators = list[ConstraintOperator] | tuple[ConstraintOperator, ...]

# List or tuple of sympy expressions
# Adding in Any here since sympy types are a bit funky at the moment
Exprs = list[Expr | Any] | tuple[Expr | Any, ...]
