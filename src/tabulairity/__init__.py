from importlib.metadata import PackageNotFoundError, version

from . import core
from . import scrapertools
from . import gsheetconnector
from . import selfimprovement

from .core import *

try:
    __version__ = version("TabulAIrity")
except PackageNotFoundError:
    __version__ = "1.3.0"