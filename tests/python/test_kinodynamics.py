"""
Test function related to center of mass residuals
"""

import aligator
import numpy as np
import pinocchio as pin

from aligator import manifolds


model = pin.buildSampleModelHumanoid()
rdata: pin.Data = model.createData()
gravity = np.array([0, 0, -9.81])
np.random.seed(0)
FD_EPS = 1e-8
THRESH = 2 * FD_EPS**0.5
sample_factor = 0.1

nq = model.nq
nv = model.nv
nu = model.nv


def sample_gauss(space):
    x0 = space.neutral()
    d = np.random.randn(space.ndx) * 0.1
    x1 = space.integrate(x0, d)
    return x0, d, x1


def test_centroidal_momentum():
    space = manifolds.MultibodyPhaseSpace(model)
    nk = 3
    nu = 3 * nk + model.nv - 6

    x, d, x0 = sample_gauss(space)
    u0 = np.random.randn(nu)
    contact_states = [True, False, True]

    contact_ids = [
        model.getFrameId("lleg_effector_body"),
        model.getFrameId("rleg_effector_body"),
        model.getFrameId("rarm_effector_body"),
    ]

    fun = aligator.CentroidalMomentumDerivativeResidual(
        space.ndx, model, gravity, contact_states, contact_ids
    )
    fdata = fun.createData()

    fun_fd = aligator.FiniteDifferenceHelper(space, fun, FD_EPS)
    fdata2 = fun_fd.createData()
    fun.evaluate(x0, u0, x0, fdata)
    fun_fd.evaluate(x0, u0, x0, fdata2)
    assert np.allclose(fdata.value, fdata2.value)

    fun_fd.computeJacobians(x0, u0, x0, fdata2)
    J_fd = fdata2.Jx
    J_fd_u = fdata2.Ju
    assert fdata.Jx.shape == J_fd.shape
    assert fdata.Ju.shape == J_fd_u.shape

    for i in range(100):
        du = np.random.randn(nu) * sample_factor
        u1 = u0 + du
        x, d, x0 = sample_gauss(space)
        fun.evaluate(x0, u1, x0, fdata)
        fun.computeJacobians(x0, u1, x0, fdata)
        fun_fd.evaluate(x0, u1, x0, fdata2)
        fun_fd.computeJacobians(x0, u1, x0, fdata2)
        assert np.linalg.norm(fdata.Ju - fdata2.Ju) <= THRESH
        assert np.linalg.norm(fdata.Jx - fdata2.Jx) <= THRESH


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main(sys.argv))
