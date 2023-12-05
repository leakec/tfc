import sys

if sys.version_info >= (3, 9):
    Tuple = tuple
else:
    from typing import Tuple

__version__ = "1.0.11"


def _version_as_tuple(version_str: str) -> Tuple[int, ...]:
    return tuple(int(i) for i in version_str.split(".") if i.isdigit())


__version_info__ = _version_as_tuple(__version__)
