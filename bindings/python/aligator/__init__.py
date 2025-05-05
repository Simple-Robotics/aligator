"""
Copyright (C) 2022 LAAS-CNRS, 2022-2025 INRIA
"""

from .pyaligator import *
from .pyaligator import __version__
from . import utils

from proxsuite_nlp import constraints


def _process():
    import sys
    import inspect
    from . import pyaligator

    lib_name = "aligator"
    submodules = inspect.getmembers(pyaligator, inspect.ismodule)
    for mod_info in submodules:
        mod_name = "{}.{}".format(lib_name, mod_info[0])
        sys.modules[mod_name] = mod_info[1]
        mod_info[1].__file__ = pyaligator.__file__
        mod_info[1].__name__ = mod_name
    sys.modules["{}.constraints".format(lib_name)] = constraints


_process()
