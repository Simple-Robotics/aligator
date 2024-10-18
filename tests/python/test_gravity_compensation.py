import aligator
import numpy as np
import pinocchio as pin

from aligator import GravityCompensationResidual, GravityCompensationData


model = pin.buildSampleModelManipulator()
rdata = model.createData()
np.set_printoptions(precision=3)
nq = model.nq
nv = model.nv
EPS = 1e-7


def test_gravity_comp_configuration():
    space = aligator.manifolds.MultibodyConfiguration(model)
    ndx = space.ndx
    assert nv == ndx
    res = GravityCompensationResidual(ndx, model)
    assert res.nr == nv
    assert res.nu == nv
    assert res.ndx1 == ndx

    data = res.createData()
    assert isinstance(data, GravityCompensationData)
    x0 = space.rand()
    x0 = np.clip(x0, -10, 10)
    u0 = np.zeros(nv)
    res.evaluate(x0, u0, data)
    res.computeJacobians(x0, u0, data)

    assert np.allclose(-data.value, pin.computeGeneralizedGravity(model, rdata, x0))
    assert np.allclose(data.Ju, np.eye(nv))
    assert np.allclose(
        -data.Jx, pin.computeGeneralizedGravityDerivatives(model, rdata, x0)
    )

    # make a cost
    weight = np.eye(res.nr)
    cost = aligator.QuadraticResidualCost(space, res, weight)
    cdata = cost.createData()
    cost.evaluate(x0, u0, cdata)


def test_gravity_comp_phase():
    space = aligator.manifolds.MultibodyPhaseSpace(model)
    ndx = space.ndx
    res = GravityCompensationResidual(ndx, model)

    assert res.nr == nv
    assert res.nu == nv
    assert res.ndx1 == ndx

    data = res.createData()
    x0 = space.rand()
    x0 = np.clip(x0, -10, 10)
    q0 = x0[:nq]
    u0 = np.zeros(nv)
    res.evaluate(x0, u0, data)
    res.computeJacobians(x0, u0, data)
    print(data.value)
    assert np.allclose(-data.value, pin.computeGeneralizedGravity(model, rdata, q0))
    assert np.allclose(data.Ju, np.eye(nv))
    Jq = data.Jx[:, :nv]
    assert np.allclose(-Jq, pin.computeGeneralizedGravityDerivatives(model, rdata, q0))


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main(sys.argv))
