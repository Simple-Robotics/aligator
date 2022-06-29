"""
Create continuous dynamics from Python.
"""
import pytest
import numpy as np
from proxddp import dynamics, manifolds


def test_abstract():
    """Test have the right types, etc."""
    space = manifolds.SO3()
    nu = 1
    cm = dynamics.ContinuousDynamicsBase(space, nu)
    cd = cm.createData()
    assert isinstance(cd, dynamics.ContinuousDynamicsData)
    assert hasattr(cd, "Jx")
    assert hasattr(cd, "Ju")
    assert hasattr(cd, "Jxdot")

    ode = dynamics.ODEAbstract(space, nu)
    ode_data = ode.createData()
    assert isinstance(ode_data, dynamics.ODEData)
    assert hasattr(ode_data, "xdot")


def test_multibody_free():
    import pinocchio as pin
    model = pin.buildSampleModelHumanoid()
    space = manifolds.MultibodyPhaseSpace(model)
    B = np.eye(model.nv)
    ode = dynamics.MultibodyFreeFwdDynamics(space, B)
    data = ode.createData()
    assert hasattr(data, "tau")


if __name__ == '__main__':
    import sys
    retcode = pytest.main(sys.argv)
