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
    dae = dynamics.ContinuousDynamicsBase(space, nu)
    dae_data = dae.createData()
    assert isinstance(dae_data, dynamics.ContinuousDynamicsData)
    assert hasattr(dae_data, "Jx")
    assert hasattr(dae_data, "Ju")
    assert hasattr(dae_data, "Jxdot")

    ode = dynamics.ODEAbstract(space, nu)
    ode_data = ode.createData()
    assert isinstance(ode_data, dynamics.ODEData)
    assert hasattr(ode_data, "xdot")


def test_multibody_free():
    try:
        import pinocchio as pin

        model = pin.buildSampleModelHumanoid()
        space = manifolds.MultibodyPhaseSpace(model)
        nu = model.nv
        B = np.eye(nu)
        ode = dynamics.MultibodyFreeFwdDynamics(space, B)
        data = ode.createData()
        assert isinstance(data, dynamics.MultibodyFreeFwdData)
        assert hasattr(data, "tau")

        x0 = space.neutral()
        u0 = np.random.randn(nu)

        ode.forward(x0, u0, data)
        ode.dForward(x0, u0, data)
    except ImportError:
        pass


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
