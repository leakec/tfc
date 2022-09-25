import sys
from typing import Union, Any
import numpy as np
import numpy.typing as npt

if sys.version_info >= (3, 8):
    from typing import Literal

    Literal = Literal
else:
    from typing_extensions import Literal

    Literal = Literal

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

# Array-like of strings
from numpy._typing._array_like import _ArrayLikeStr_co

StrArrayLike = _ArrayLikeStr_co
npt.ArrayLike
