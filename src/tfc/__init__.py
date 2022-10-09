from . import utils
from .utfc import utfc
from .utfc import HybridUtfc
from .mtfc import mtfc

from .version import __version__ as __version__
from .version import __version_info__ as __version_info__

__all__ = ["utfc", "mtfc", "utils", "HybridUtfc"]

# import sys
# sys.ps1 = "TFC > "
# sys.ps2 = "..... "
