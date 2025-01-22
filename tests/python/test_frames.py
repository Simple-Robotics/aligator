"""
Test function related to Pinocchio frames.
This test is not added to CTest when CMake does not detect Pinocchio.
"""

import aligator
from aligator import manifolds

import numpy as np
import pinocchio as pin
import pytest


model = pin.buildSampleModelHumanoid()
rdata: pin.Data = model.createData()
SEED = 0
np.random.seed(SEED)
aligator.seed(SEED)
FD_EPS = 1e-7
ATOL = 2 * FD_EPS**0.5

nq = model.nq
nv = model.nv
nu = model.nv


def sample_gauss(space):
    x0 = space.neutral()
    d = np.random.randn(space.ndx) * 0.1
    x0[:] = space.integrate(x0, d)
    return x0


def test_frame_placement():
    fr_name1 = "larm_shoulder2_body"
    fr_id1 = model.getFrameId(fr_name1)

    space = manifolds.MultibodyConfiguration(model)
    ndx = space.ndx
    x0 = space.neutral()
    d = np.random.randn(space.ndx) * 0.1
    d[6:] = 0.0
    x0 = space.integrate(x0, d)
    u0 = np.zeros(nu)
    q0 = x0[:nq]

    pin.framesForwardKinematics(model, rdata, q0)
    fr_plc1 = rdata.oMf[fr_id1]

    fun = aligator.FramePlacementResidual(ndx, nu, model, fr_plc1, fr_id1)
    assert fr_id1 == fun.frame_id
    assert fr_plc1 == fun.getReference()

    fdata = fun.createData()
    fun.evaluate(x0, fdata)

    assert np.allclose(fdata.value, 0.0)

    fun.computeJacobians(x0, fdata)
    J = fdata.Jx[:, :nv]

    pin.computeJointJacobians(model, rdata)
    realJ = pin.getFrameJacobian(model, rdata, fr_id1, pin.LOCAL)
    assert J.shape == realJ.shape
    assert np.allclose(fdata.Jx[:, :nv], realJ)

    fun_fd = aligator.FiniteDifferenceHelper(space, fun, FD_EPS)
    fdata2 = fun_fd.createData()
    fun_fd.evaluate(x0, u0, fdata2)
    assert np.allclose(fdata.value, fdata2.value)

    fun_fd.computeJacobians(x0, u0, fdata2)
    J_fd = fdata2.Jx[:]
    assert fdata.Jx.shape == J_fd.shape

    for i in range(100):
        x0 = sample_gauss(space)
        fun.evaluate(x0, u0, fdata)
        fun.computeJacobians(x0, u0, fdata)
        fun_fd.evaluate(x0, u0, fdata2)
        fun_fd.computeJacobians(x0, u0, fdata2)
        assert np.allclose(fdata.Jx, fdata2.Jx, ATOL)


def test_frame_translation():
    fr_name1 = "larm_shoulder2_body"
    fr_id1 = model.getFrameId(fr_name1)

    space = manifolds.MultibodyConfiguration(model)
    ndx = space.ndx
    x0 = space.neutral()
    d = np.random.randn(space.ndx) * 0.1
    d[6:] = 0.0
    x0 = space.integrate(x0, d)
    u0 = np.zeros(nu)
    q0 = x0[:nq]

    pin.framesForwardKinematics(model, rdata, q0)
    target_pos = rdata.oMf[fr_id1].translation

    fun = aligator.FrameTranslationResidual(ndx, nu, model, target_pos, fr_id1)
    assert fr_id1 == fun.frame_id

    fdata = fun.createData()
    fun.evaluate(x0, u0, fdata)

    assert np.allclose(fdata.value, 0.0)

    fun.computeJacobians(x0, u0, fdata)

    fun_fd = aligator.FiniteDifferenceHelper(space, fun, FD_EPS)
    fdata2 = fun_fd.createData()
    fun_fd.evaluate(x0, u0, fdata2)
    fun_fd.computeJacobians(x0, u0, fdata2)
    # just check dims for now
    assert fdata.Jx.shape == fdata2.Jx.shape

    for i in range(100):
        x0 = sample_gauss(space)
        fun.evaluate(x0, fdata)
        fun.computeJacobians(x0, fdata)
        fun_fd.evaluate(x0, u0, fdata2)
        fun_fd.computeJacobians(x0, u0, fdata2)
        assert np.allclose(fdata.Jx, fdata2.Jx, atol=ATOL)


def test_frame_velocity():
    fr_name1 = "larm_shoulder2_body"
    fr_id1 = model.getFrameId(fr_name1)
    space = manifolds.MultibodyPhaseSpace(model)

    x0 = sample_gauss(space)
    q0, v0 = x0[: model.nq], x0[model.nq :]
    u0 = np.zeros(nu)

    pin.forwardKinematics(model, rdata, q0, v0)
    ref_type = pin.LOCAL
    v_ref = pin.getFrameVelocity(model, rdata, fr_id1, ref_type)

    fun = aligator.FrameVelocityResidual(space.ndx, nu, model, v_ref, fr_id1, ref_type)
    assert fr_id1 == fun.frame_id
    assert np.allclose(v_ref, fun.getReference())

    fdata = fun.createData()
    fun.evaluate(x0, fdata)
    print("v_ref=", v_ref)

    assert np.allclose(fdata.value, 0.0)

    fun.evaluate(x0, fdata)
    fun.computeJacobians(x0, fdata)

    fun_fd = aligator.FiniteDifferenceHelper(space, fun, FD_EPS)
    fdata2 = fun_fd.createData()
    fun_fd.evaluate(x0, u0, fdata2)
    fun_fd.computeJacobians(x0, u0, fdata2)
    assert fdata.Jx.shape == fdata2.Jx.shape

    for i in range(100):
        x0 = sample_gauss(space)
        fun.evaluate(x0, fdata)
        fun.computeJacobians(x0, fdata)
        fun_fd.evaluate(x0, u0, fdata2)
        fun_fd.computeJacobians(x0, u0, fdata2)
        assert np.allclose(fdata.Jx, fdata2.Jx, atol=ATOL)


def test_fly_high():
    fr_name1 = "larm_shoulder2_body"
    fr_id1 = model.getFrameId(fr_name1)
    space = manifolds.MultibodyPhaseSpace(model)
    fun = aligator.FlyHighResidual(space.ndx, model, fr_id1, 0.1, nu)
    data = fun.createData()
    data2 = fun.createData()
    Jx_nd = data.Jx.copy()
    np.set_printoptions(precision=2, linewidth=250)

    for _ in range(10):
        x0 = sample_gauss(space)
        fun.evaluate(x0, data)
        fun.computeJacobians(x0, data)

        ei = np.zeros(space.ndx)
        for i in range(space.ndx):
            ei[i] = FD_EPS
            x0_p = space.integrate(x0, ei)
            fun.evaluate(x0_p, data2)
            Jx_nd[:, i] = (data2.value - data.value) / FD_EPS
            ei[i] = 0.0

        err_Jx = data.Jx - Jx_nd
        print(err_Jx, err_Jx.max())
        assert np.allclose(data.Jx, Jx_nd, atol=ATOL)


def test_frame_collision():
    import coal

    fr_name1 = "larm_shoulder2_body"
    fr_id1 = model.getFrameId(fr_name1)
    joint_id = model.frames[fr_id1].parentJoint

    fr_name2 = "rleg_elbow_body"
    fr_id2 = model.getFrameId(fr_name2)
    joint_id2 = model.frames[fr_id2].parentJoint

    frame_SE3 = pin.SE3.Random()
    frame_SE3_bis = pin.SE3.Random()
    alpha = np.random.rand()
    beta = np.random.rand()
    gamma = np.random.rand()
    delta = np.random.rand()

    geometry = pin.buildSampleGeometryModelHumanoid(model)
    ig_frame = geometry.addGeometryObject(
        pin.GeometryObject(
            "frame", fr_id1, joint_id, coal.Capsule(alpha, gamma), frame_SE3
        )
    )

    ig_frame2 = geometry.addGeometryObject(
        pin.GeometryObject(
            "frame2", fr_id2, joint_id2, coal.Capsule(beta, delta), frame_SE3_bis
        )
    )
    geometry.addCollisionPair(pin.CollisionPair(ig_frame, ig_frame2))
    gdata = geometry.createData()

    space = manifolds.MultibodyConfiguration(model)
    ndx = space.ndx
    x0 = space.neutral()
    d = np.random.randn(space.ndx) * 0.1
    d[6:] = 0.0
    x0 = space.integrate(x0, d)
    u0 = np.zeros(nu)
    q0 = x0[:nq]

    pin.forwardKinematics(model, rdata, q0)
    pin.updateGeometryPlacements(model, rdata, geometry, gdata, q0)
    pin.computeDistance(geometry, gdata, 0)
    norm = gdata.distanceResults[0].min_distance

    fun = aligator.FrameCollisionResidual(ndx, nu, model, geometry, 0)

    fdata = fun.createData()
    fun.evaluate(x0, fdata)

    assert np.allclose(fdata.value[0], norm)

    fun.computeJacobians(x0, fdata)
    fun_fd = aligator.FiniteDifferenceHelper(space, fun, FD_EPS)
    fdata2 = fun_fd.createData()
    fun_fd.evaluate(x0, u0, fdata2)
    assert np.allclose(fdata.value, fdata2.value)

    fun_fd.computeJacobians(x0, u0, fdata2)
    J_fd = fdata2.Jx[:]
    assert fdata.Jx.shape == J_fd.shape

    for i in range(100):
        x0 = sample_gauss(space)
        fun.evaluate(x0, u0, fdata)
        fun.computeJacobians(x0, u0, fdata)
        fun_fd.evaluate(x0, u0, fdata2)
        fun_fd.computeJacobians(x0, u0, fdata2)
        assert np.allclose(fdata.Jx, fdata2.Jx, atol=ATOL)


def test_frame_collision_no_collision_pairs():
    space = manifolds.MultibodyConfiguration(model)
    ndx = space.ndx
    x0 = space.neutral()
    d = np.random.randn(space.ndx) * 0.1
    d[6:] = 0.0
    x0 = space.integrate(x0, d)

    geometry = pin.buildSampleGeometryModelHumanoid(model)
    with pytest.raises(IndexError, match="Provided collision pair index"):
        aligator.FrameCollisionResidual(ndx, nu, model, geometry, 0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
