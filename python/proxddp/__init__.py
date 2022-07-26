"""
Copyright (C) 2022 LAAS-CNRS, INRIA
"""
from .pyproxddp import *
from . import utils

from proxnlp import constraints, manifolds, VerboseLevel, LinesearchStrategy


def _process():
    import sys
    import inspect
    from . import pyproxddp

    lib_name = "proxddp"
    submodules = inspect.getmembers(pyproxddp, inspect.ismodule)
    for mod_info in submodules:
        mod_name = "{}.{}".format(lib_name, mod_info[0])
        sys.modules[mod_name] = mod_info[1]
        mod_info[1].__file__ = pyproxddp.__file__
        mod_info[1].__name__ = mod_name


_process()
