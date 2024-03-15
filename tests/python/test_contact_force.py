"""
Test function related to contact force
"""

import pytest

import numpy as np

import pinocchio as pin

import aligator

from test_constrained_dynamics import createFourBarLinkages
from utils import configure_functions

np.random.seed(0)
FD_EPS = 1e-8
THRESH = 2 * FD_EPS**0.5


def sample_gauss(space):
    x0 = space.neutral()
    d = np.random.randn(space.ndx) * 0.1
    x1 = space.integrate(x0, d)
    return x0, d, x1


def test_fun():
    model, constraint_model = createFourBarLinkages()
    constraint_models = [constraint_model]
    space = aligator.manifolds.MultibodyPhaseSpace(model)
    prox = pin.ProximalSettings(1e-12, 1e-10, 3)

    # Configure MultibodyConstraintCommonBuilder and add it tho a CommonModelBuilderContainer
    common_builders = aligator.CommonModelBuilderContainer()
    multibody_common = aligator.dynamics.MultibodyConstraintCommonBuilder()
    multibody_common = common_builders.get(
        "aligator::dynamics::MultibodyConstraintCommonTpl<double>", multibody_common
    )
    multibody_common.withRunAba(True)
    multibody_common.withProxSettings(prox)
    multibody_common.withPinocchioModel(model)
    multibody_common.withConstraintModels(constraint_models)

    #
    contact_ref = aligator.ContactReference(constraint_models, 0)
    fun = aligator.ContactForceResidual(model.nq + model.nv, model.nv, contact_ref)
    fun.setReference(np.array([0.0, 0.0, 0.0]))
    common_containers = configure_functions([fun], common_builders)
    fdata = fun.createData(common_containers.datas)
    assert isinstance(fdata, aligator.ContactForceData)

    nu = model.nv
    x0 = space.neutral()
    x0[: model.nq] = model.q_init
    u0 = np.random.randn(nu)

    common_containers.evaluate(x0, u0)
    fun.evaluate(x0, u0, x0, fdata)

    common_containers.compute_gradients(x0, u0)
    fun.computeJacobians(x0, u0, x0, fdata)

    fun_fd = aligator.FiniteDifferenceHelper(space, fun, FD_EPS)
    fdata2 = fun_fd.createData(common_containers.datas)
    common_containers.evaluate(x0, u0)
    fun_fd.evaluate(x0, u0, x0, fdata2)
    assert np.allclose(fdata.value, fdata2.value)

    common_containers.compute_gradients(x0, u0)
    fun_fd.computeJacobians(x0, u0, x0, fdata2)
    J_fd = fdata2.Jx[:]
    assert fdata.Jx.shape == J_fd.shape

    for i in range(100):
        x, d, x0 = sample_gauss(space)
        common_containers.evaluate(x0, u0)
        common_containers.compute_gradients(x0, u0)
        fun.evaluate(x0, u0, x0, fdata)
        fun.computeJacobians(x0, u0, x0, fdata)
        fun_fd.evaluate(x0, u0, x0, fdata2)
        fun_fd.computeJacobians(x0, u0, x0, fdata2)
        assert np.allclose(fdata.Jx, fdata2.Jx, atol=THRESH)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
