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
else:
    from typing_extensions import Annotated

from annotated_types import Gt

uint = Annotated[int, Gt(0)]
Number = Union[int, float, complex]
