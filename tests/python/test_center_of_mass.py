"""
Test function related to center of mass residuals
"""

import aligator
import numpy as np
import pinocchio as pin

from aligator import manifolds


model = pin.buildSampleModelHumanoid()
rdata: pin.Data = model.createData()
np.random.seed(0)
EPS = 1e-7
ATOL = 2 * EPS**0.5

nq = model.nq
nv = model.nv
nu = model.nv


def sample_gauss(space):
    x0 = space.neutral()
    d = np.random.randn(space.ndx) * 0.1
    x1 = space.integrate(x0, d)
    return x0, d, x1


def test_com_placement():
    space = manifolds.MultibodyConfiguration(model)
    ndx = space.ndx
    x0 = space.neutral()
    dx = np.random.randn(space.ndx) * 0.1
    dx[6:] = 0.0
    x0 = space.integrate(x0, dx)
    u0 = np.zeros(nu)
    q0 = x0[:nq]

    pin.centerOfMass(model, rdata, q0)
    com_plc1 = rdata.com[0]
    fun = aligator.CenterOfMassTranslationResidual(ndx, nu, model, com_plc1)
    assert np.allclose(com_plc1, fun.getReference())

    fdata = fun.createData()
    fun.evaluate(x0, fdata)

    assert np.allclose(fdata.value, 0.0)

    fun.computeJacobians(x0, fdata)
    J = fdata.Jx[:, :nv]

    pin.computeJointJacobians(model, rdata)
    realJ = pin.jacobianCenterOfMass(model, rdata, q0)
    assert J.shape == realJ.shape
    assert np.allclose(fdata.Jx[:, :nv], realJ)

    fun_fd = aligator.FiniteDifferenceHelper(space, fun, EPS)
    fdata2 = fun_fd.createData()
    fun_fd.evaluate(x0, u0, x0, fdata2)
    assert np.allclose(fdata.value, fdata2.value)

    fun_fd.computeJacobians(x0, u0, x0, fdata2)
    J_fd = fdata2.Jx[:]
    assert fdata.Jx.shape == J_fd.shape

    for i in range(100):
        x, d, x0 = sample_gauss(space)
        fun.evaluate(x0, u0, x0, fdata)
        fun.computeJacobians(x0, u0, x0, fdata)
        fun_fd.evaluate(x0, u0, x0, fdata2)
        fun_fd.computeJacobians(x0, u0, x0, fdata2)
        assert np.allclose(fdata.Jx, fdata2.Jx, ATOL)


def test_frame_velocity():
    # rdata: pin.Data = model.createData()

    space = manifolds.MultibodyPhaseSpace(model)
    x, d, x0 = sample_gauss(space)
    q0, v0 = x0[: model.nq], x0[model.nq :]
    u0 = np.zeros(nu)

    pin.centerOfMass(model, rdata, q0, v0)
    com_vel1 = rdata.vcom[0]

    fun = aligator.CenterOfMassVelocityResidual(space.ndx, nu, model, com_vel1)
    assert np.allclose(com_vel1, fun.getReference())

    fdata = fun.createData()
    fun.evaluate(x0, fdata)

    assert np.allclose(fdata.value, 0.0)

    fun.computeJacobians(x0, fdata)

    fun_fd = aligator.FiniteDifferenceHelper(space, fun, EPS)
    fdata2 = fun_fd.createData()
    fun_fd.evaluate(x0, u0, x0, fdata2)
    fun_fd.computeJacobians(x0, u0, x0, fdata2)
    assert fdata.Jx.shape == fdata2.Jx.shape

    for i in range(100):
        x, d, x0 = sample_gauss(space)
        fun.evaluate(x0, fdata)
        fun.computeJacobians(x0, fdata)
        fun_fd.evaluate(x0, u0, x0, fdata2)
        fun_fd.computeJacobians(x0, u0, x0, fdata2)
        print(i)
        print(fdata.Jx)
        print(fdata2.Jx)
        print(fdata.Jx)
        assert np.allclose(fdata.Jx, fdata2.Jx, ATOL)


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main(sys.argv))
