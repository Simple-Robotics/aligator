"""
Create continuous dynamics from Python.
"""
from proxddp import dynamics
from proxnlp.manifolds import SE2


space = SE2()
nu = space.ndx

# instantiate base object
dynbase = dynamics.ContinuousDynamicsBase(space, nu)
dyndata = dynbase.createData()
