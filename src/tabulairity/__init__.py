from importlib.metadata import PackageNotFoundError, version

from . import core
from . import scrapertools
from . import gsheetconnector
from . import selfimprovement
try:
    from .visualization import vizOn, vizOff
except Exception:
    # viz is optional — core still works without it
    def vizOn(*a, **kw):
        print("[Viz] visualization module not available")
        return None
    def vizOff(*a, **kw):
        pass

from .core import *

try:
    __version__ = version("TabulAIrity")
except PackageNotFoundError:
    __version__ = "1.3.0"