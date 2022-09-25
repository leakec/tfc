import os
import sys
from typing import Union, Any
from numpy import typing as npt
import numpy as np

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
else:
    from typing_extensions import Annotated
    from typing import List, Tuple, Dict

from annotated_types import Gt, Ge

# Path
#Path = Union[str, os.PathLike]
Path = str

# Integer > 0
pint = Annotated[int, Gt(0)]

# Integer >= 0
uint = Annotated[int, Ge(0)]

# General number type
Number = Union[int, float, complex]

# Array-like of strings
StrArrayLike = np._typing._array_like._ArrayLikeStr_co

