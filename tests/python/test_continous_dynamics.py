"""
Create continuous dynamics from Python.
"""
import numpy as np
from proxddp import dynamics, manifolds


def test_dae():

    space = manifolds.SE2()
    nu = space.ndx

test_dae()