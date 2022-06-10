"""
Create continuous dynamics from Python.
"""
import numpy as np
from proxddp import dynamics


def test_dae():
    from proxnlp.manifolds import SE2

    space = SE2()
    nu = space.ndx

    # instantiate base object
    dynbase = dynamics.ContinuousDynamicsBase(space, nu)
    dyndata = dynbase.createData()
    x0 = space.rand()
    x1 = space.rand()
    u0 = np.random.randn(nu)
    dynbase.evaluate(x0, u0, x1, dyndata)
