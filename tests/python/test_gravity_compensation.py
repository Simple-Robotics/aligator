import aligator
import numpy as np
import pytest

HAS_PINOCCHIO = aligator.has_pinocchio_features()
pytestmark = pytest.mark.skipif(
    not HAS_PINOCCHIO, reason="Aligator was compiled without Pinocchio."
)

np.set_printoptions(precision=3)
EPS = 1e-7


@pytest.fixture
def sample_model():
    import pinocchio as pin

    model = pin.buildSampleModelManipulator()
    rdata = model.createData()
    return model, rdata


def test_gravity_comp_configuration(sample_model):
    from aligator import GravityCompensationResidual, GravityCompensationData
    import pinocchio as pin

    model, rdata = sample_model

    space = aligator.manifolds.MultibodyConfiguration(model)
    ndx = space.ndx
    assert model.nv == ndx
    res = GravityCompensationResidual(ndx, model)
    assert res.nr == model.nv
    assert res.nu == model.nv
    assert res.ndx1 == ndx

    data = res.createData()
    assert isinstance(data, GravityCompensationData)
    x0 = space.rand()
    x0 = np.clip(x0, -10, 10)
    u0 = np.zeros(model.nv)
    res.evaluate(x0, u0, data)
    res.computeJacobians(x0, u0, data)

    assert np.allclose(-data.value, pin.computeGeneralizedGravity(model, rdata, x0))
    assert np.allclose(data.Ju, np.eye(model.nv))
    assert np.allclose(
        -data.Jx, pin.computeGeneralizedGravityDerivatives(model, rdata, x0)
    )

    # make a cost
    weight = np.eye(res.nr)
    cost = aligator.QuadraticResidualCost(space, res, weight)
    cdata = cost.createData()
    cost.evaluate(x0, u0, cdata)


def test_gravity_comp_phase(sample_model):
    from aligator import GravityCompensationResidual
    import pinocchio as pin

    model, rdata = sample_model

    space = aligator.manifolds.MultibodyPhaseSpace(model)
    ndx = space.ndx
    res = GravityCompensationResidual(ndx, model)

    assert res.nr == model.nv
    assert res.nu == model.nv
    assert res.ndx1 == ndx

    data = res.createData()
    x0 = space.rand()
    x0 = np.clip(x0, -10, 10)
    q0 = x0[: model.nq]
    u0 = np.zeros(model.nv)
    res.evaluate(x0, u0, data)
    res.computeJacobians(x0, u0, data)
    print(data.value)
    assert np.allclose(-data.value, pin.computeGeneralizedGravity(model, rdata, q0))
    assert np.allclose(data.Ju, np.eye(model.nv))
    Jq = data.Jx[:, : model.nv]
    assert np.allclose(-Jq, pin.computeGeneralizedGravityDerivatives(model, rdata, q0))


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main(sys.argv))
