"""
Car parking example from Tassa et al. 2014.

TODO: finish this
"""
import numpy as np
import proxddp
import pinocchio as pin
from proxddp import dynamics, manifolds
import tap


class Args(tap.Tap):
    pass


space = manifolds.TSE2()

print(space)
ndx = space.ndx

seed = 42
np.random.seed(seed)
pin.seed(seed)
x0 = space.rand()

# Control model:
