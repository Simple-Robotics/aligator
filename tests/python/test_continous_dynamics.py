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
    nu = model.nv
    B = np.eye(nu)
    ode = dynamics.MultibodyFreeFwdDynamics(space, B)
    data = ode.createData()
    assert hasattr(data, "tau")

    x0 = space.neutral()
    u0 = np.random.randn(nu)

    ode.forward(x0, u0, data)
    ode.dForward(x0, u0, data)

    # compare with croc
    import crocoddyl as cc
    state_ = cc.StateMultibody(model)
    actuation_ = cc.ActuationModelFull(state_)
    cost_ = cc.CostModelSum(state_, nu)
    cm = cc.DifferentialActionModelFreeFwdDynamics(state_, actuation_, cost_)
    cm_data = cm.createData()
    cm.calc(cm_data, x0, u0)
    cm.calcDiff(cm_data, x0, u0)

    err_x = abs(cm_data.Fx - data.Jx[model.nv:])
    err_u = abs(cm_data.Fu - data.Ju[model.nv:])
    assert np.allclose(err_x, 0.)
    assert np.allclose(err_u, 0.)



if __name__ == '__main__':
    import sys
    retcode = pytest.main(sys.argv)
