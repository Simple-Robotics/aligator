"""
Copyright (C) 2022 LAAS-CNRS, INRIA
"""
from .pyproxddp import *
from .pyproxddp import __version__
from . import utils

from proxsuite_nlp import (
    constraints,
    manifolds,
)


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
    sys.modules["{}.manifolds".format(lib_name)] = manifolds
    sys.modules["{}.constraints".format(lib_name)] = constraints


_process()
