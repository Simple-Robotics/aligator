"""
Copyright (C) 2022 LAAS-CNRS, INRIA
"""
import proxnlp

from .pyproxddp import *
from . import utils

from proxnlp import constraints, manifolds, VerboseLevel


def _process():
    import sys
    import inspect
    from . import pyproxddp

    lib_name = "proxddp"
    submodules = inspect.getmembers(pyproxddp, inspect.ismodule)
    for mod_info in submodules:
        sys.modules["{}.{}".format(lib_name, mod_info[0])] = mod_info[1]


_process()
