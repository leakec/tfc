import os
import sys
from typing import Union

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
else:
    from typing_extensions import Annotated
    from typing import List, Tuple

from annotated_types import Gt

Path = Union[str, os.PathLike]

uint = Annotated[int, Gt(0)]
Number = Union[int, float, complex]
