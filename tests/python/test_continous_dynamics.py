"""
Create continuous dynamics from Python.
"""
import pytest
import numpy as np
from proxddp import dynamics, manifolds


def test_abstract():
    space = manifolds.SO3()
    nu = 1
    cm = dynamics.ContinuousDynamicsBase(space, nu)
    cd = cm.createData()
    print(cd)

    ode = dynamics.ODEAbstract(space, nu)
    ode_data = ode.createData()
    print(ode_data)
    assert isinstance(ode_data, dynamics.ODEData)


def test_multibody_free():
    import pinocchio as pin
    model = pin.buildSampleModelHumanoid()
    space = manifolds.MultibodyPhaseSpace(model)
    B = np.eye(model.nv)
    ode = dynamics.MultibodyFreeFwdDynamics(space, B)
    data = ode.createData()
    assert hasattr(data, "tau")


def test_dae():
    space = manifolds.SE2()
    nu = space.ndx


if __name__ == '__main__':
    import sys
    retcode = pytest.main(sys.argv)
