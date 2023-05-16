"""
Test function related to frames.
"""
import proxddp
import numpy as np
import pinocchio as pin

from proxddp import manifolds


model = pin.buildSampleModelHumanoid()
rdata: pin.Data = model.createData()
FD_EPS = 1e-4
THRESH = 1e-2

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

    fun = proxddp.FramePlacementResidual(ndx, nu, model, fr_plc1, fr_id1)
    assert fr_id1 == fun.frame_id
    assert fr_plc1 == fun.getReference()

    fdata = fun.createData()
    fun.evaluate(x0, u0, x0, fdata)

    assert np.allclose(fdata.value, 0.0)
    print(fdata.value)

    print("Error:", fdata.rMf)

    fun.computeJacobians(x0, u0, x0, fdata)
    J = fdata.Jx[:, :nv]

    pin.computeJointJacobians(model, rdata)
    realJ = pin.getFrameJacobian(model, rdata, fr_id1, pin.LOCAL)
    assert J.shape == realJ.shape
    assert np.allclose(fdata.Jx[:, :nv], realJ)

    fun_fd = proxddp.FiniteDifferenceHelper(space, fun, FD_EPS)
    fdata2 = fun_fd.createData()
    fun_fd.evaluate(x0, u0, x0, fdata2)
    fun_fd.computeJacobians(x0, u0, x0, fdata2)
    J_fd = fdata2.Jx[:]
    assert fdata.Jx.shape == J_fd.shape
    assert np.allclose(fdata.Jx, J_fd, THRESH)
    for i in range(100):
        x0 = space.neutral()
        d = np.random.randn(space.ndx) * 0.1
        d[6:] = 0.0
        x0 = space.integrate(x0, d)
        fun.evaluate(x0, u0, x0, fdata)
        fun.computeJacobians(x0, u0, x0, fdata)
        fun_fd.evaluate(x0, u0, x0, fdata2)
        fun_fd.computeJacobians(x0, u0, x0, fdata2)
        assert np.allclose(fdata.Jx, fdata2.Jx, THRESH)


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

    fun = proxddp.FrameTranslationResidual(ndx, nu, model, target_pos, fr_id1)
    assert fr_id1 == fun.frame_id

    fdata = fun.createData()
    fun.evaluate(x0, u0, x0, fdata)

    assert np.allclose(fdata.value, 0.0)
    print(fdata.value)
    pdata_f = fdata.pin_data
    print(pdata_f.oMf[fr_id1])

    fun.computeJacobians(x0, u0, x0, fdata)

    fun_fd = proxddp.FiniteDifferenceHelper(space, fun, FD_EPS)
    fdata2 = fun_fd.createData()
    fun_fd.evaluate(x0, u0, x0, fdata2)
    fun_fd.computeJacobians(x0, u0, x0, fdata2)
    assert fdata.Jx.shape == fdata2.Jx.shape
    assert np.allclose(fdata.Jx, fdata2.Jx, THRESH)
    for i in range(100):
        x0 = space.neutral()
        d = np.random.randn(space.ndx) * 0.1
        x0 = space.integrate(x0, d)
        fun.evaluate(x0, u0, x0, fdata)
        fun.computeJacobians(x0, u0, x0, fdata)
        fun_fd.evaluate(x0, u0, x0, fdata2)
        fun_fd.computeJacobians(x0, u0, x0, fdata2)
        assert np.allclose(fdata.Jx, fdata2.Jx, THRESH)


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

    fun = proxddp.FrameVelocityResidual(space.ndx, nu, model, v_ref, fr_id1, ref_type)
    assert fr_id1 == fun.frame_id
    assert np.allclose(v_ref, fun.getReference())

    fdata = fun.createData()
    fun.evaluate(x0, fdata)
    print("v_ref=", v_ref)

    assert np.allclose(fdata.value, 0.0)
    print("pindata from fdata:")

    fun.evaluate(x0, fdata)
    fun.computeJacobians(x0, fdata)
    J = fdata.Jx[:, :nv]
    realJ, _ = pin.getFrameVelocityDerivatives(model, rdata, fr_id1, ref_type)
    assert J.shape == realJ.shape
    assert np.allclose(fdata.Jx[:, :nv], realJ)

    fun_fd = proxddp.FiniteDifferenceHelper(space, fun, FD_EPS)
    fdata2 = fun_fd.createData()
    fun_fd.evaluate(x0, u0, x0, fdata2)
    fun_fd.computeJacobians(x0, u0, x0, fdata2)
    assert fdata.Jx.shape == fdata2.Jx.shape

    for i in range(100):
        x0 = space.neutral()
        d = np.random.randn(space.ndx) * 0.1
        x0 = space.integrate(x0, d)
        fun.evaluate(x0, fdata)
        fun.computeJacobians(x0, fdata)
        fun_fd.evaluate(x0, u0, x0, fdata2)
        fun_fd.computeJacobians(x0, u0, x0, fdata2)
        assert np.allclose(fdata.Jx, fdata2.Jx, THRESH)


def test_fly_high():
    fr_name1 = "larm_shoulder2_body"
    fr_id1 = model.getFrameId(fr_name1)
    space = manifolds.MultibodyPhaseSpace(model)
    fun = proxddp.FlyHighResidual(space, fr_id1, 0.1, nu)
    data = fun.createData()
    data2 = fun.createData()
    Jx_nd = data.Jx.copy()
    np.set_printoptions(precision=2, linewidth=250)

    for _ in range(10):
        x0 = space.neutral()
        d = np.random.randn(space.ndx) * 0.1
        x0 = space.integrate(x0, d)

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
        print(err_Jx)
        assert np.allclose(data.Jx, Jx_nd, THRESH)


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main(sys.argv))
