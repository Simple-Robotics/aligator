"""
Test function related to frames.
"""
import proxddp
import numpy as np
import pinocchio as pin

from proxddp import manifolds


model = pin.buildSampleModelHumanoid()
rdata: pin.Data = model.createData()
space = manifolds.MultibodyConfiguration(model)

ndx = space.ndx
nq = model.nq
nv = model.nv
nu = model.nv


def test_frame_placement():
    fr_name1 = "larm_shoulder2_body"
    print(model.frames.tolist())
    fr_id1 = model.getFrameId(fr_name1)

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
    print("pindata from fdata:")
    pdata_f = fdata.pin_data
    print(pdata_f.oMf[fr_id1])

    fun.computeJacobians(x0, u0, x0, fdata)
    J = fdata.Jx[:, :nv]
    print("JAC:", J)

    pin.computeJointJacobians(model, rdata)
    realJ = pin.getFrameJacobian(model, rdata, fr_id1, pin.LOCAL)
    print("ACTUAL J:", realJ)
    assert J.shape == realJ.shape
    assert np.allclose(fdata.Jx[:, :nv], realJ)

    fun_fd = proxddp.FiniteDifferenceHelper(space, fun, 1e-3)
    fdata2 = fun_fd.createData()
    fun_fd.evaluate(x0, u0, x0, fdata2)
    fun_fd.computeJacobians(x0, u0, x0, fdata2)
    J_fd = fdata2.Jx[:]
    assert np.allclose(J, J_fd, 1e-2)


def test_frame_velocity():
    fr_name1 = "larm_shoulder2_body"
    print(model.frames.tolist())
    fr_id1 = model.getFrameId(fr_name1)

    x0 = space.neutral()
    d = np.random.randn(space.ndx) * 0.1
    d[6:] = 0.0
    x0 = space.integrate(x0, d)
    u0 = np.zeros(nu)

    v_ref = pin.getFrameVelocity(model, rdata, fr_id1, pin.LOCAL)

    fun = proxddp.FrameVelocityResidual(ndx, nu, model, v_ref, fr_id1, pin.LOCAL)
    assert fr_id1 == fun.frame_id
    assert v_ref == fun.getReference()

    fdata = fun.createData()
    fun.evaluate(x0, u0, x0, fdata)

    assert np.allclose(fdata.value, 0.0)
    print(fdata.value)
    print("pindata from fdata:")
    pdata_f = fdata.pin_data
    print(pdata_f.oMf[fr_id1])

    fun.computeJacobians(x0, u0, x0, fdata)
    J = fdata.Jx[:, :nv]
    print("JAC:", J)

    realJ, _ = pin.getFrameVelocityDerivatives(model, rdata, fr_id1, pin.LOCAL)
    print("ACTUAL J:", realJ)
    assert J.shape == realJ.shape
    assert np.allclose(fdata.Jx[:, :nv], realJ)


if __name__ == "__main__":
    import pytest
    import sys

    retcode = pytest.main(sys.argv)
