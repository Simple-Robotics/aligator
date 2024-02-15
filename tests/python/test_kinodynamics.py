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
    space_centroidal = manifolds.VectorSpace(6)
    space_multibody = manifolds.MultibodyPhaseSpace(model)
    nk = 3
    nu = 3 * nk + model.nv
    space = manifolds.CartesianProduct(space_centroidal, space_multibody)

    x, d, x0 = sample_gauss(space)
    u0 = np.random.randn(nu)
    contact_states = [True, False, True]
    contact_poses = [
        np.array([0.2, 0.1, 0.0]),
        np.array([0.2, 0.0, 0.0]),
        np.array([0.0, 0.1, 0.0]),
    ]
    contact_map = aligator.ContactMap(contact_states, contact_poses)

    fun = aligator.CentroidalMomentumDerivativeResidual(model, gravity, contact_map)
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


def test_wrapper_frame_placement():
    space_centroidal = manifolds.VectorSpace(6)
    space_multibody = manifolds.MultibodyPhaseSpace(model)
    space = manifolds.CartesianProduct(space_centroidal, space_multibody)

    jid = model.getFrameId("larm_effector_body")
    x, d, x0 = sample_gauss(space)
    nk = 2
    nu = nk * 3 + model.nv
    u0 = np.random.rand(nu)

    pin.framesForwardKinematics(model, rdata, x0[6 : 6 + model.nq])
    frame_ref = rdata.oMf[jid]
    frame_ref.translation[1] -= 0.05
    frame_ref.translation[2] += 0.1

    wrapped_fun = aligator.FramePlacementResidual(
        space_multibody.ndx, model.nv, model, frame_ref, jid
    )

    fun = aligator.KinodynamicsWrapperResidual(wrapped_fun, model.nq, model.nv, nk)

    fdata = fun.createData()
    wrapped_fdata = wrapped_fun.createData()
    wrapped_fun.evaluate(x0[6:], wrapped_fdata)
    fun.evaluate(x0, fdata)

    assert np.allclose(fdata.value, wrapped_fdata.value)

    fun_fd = aligator.FiniteDifferenceHelper(space, fun, FD_EPS)
    fdata2 = fun_fd.createData()
    fun_fd.evaluate(x0, u0, x0, fdata2)
    assert np.allclose(fdata.value, fdata2.value)

    fun_fd.computeJacobians(x0, u0, x0, fdata2)
    J_fd = fdata2.Jx
    J_fd_u = fdata2.Ju
    assert fdata.Jx.shape == J_fd.shape
    assert fdata.Ju.shape == J_fd_u.shape

    for i in range(100):
        x, d, x1 = sample_gauss(space)
        fun.evaluate(x1, fdata)
        fun.computeJacobians(x1, fdata)
        fun_fd.evaluate(x1, u0, x1, fdata2)
        fun_fd.computeJacobians(x1, u0, x1, fdata2)
        assert np.allclose(fdata.Jx, fdata2.Jx, THRESH)


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main(sys.argv))
