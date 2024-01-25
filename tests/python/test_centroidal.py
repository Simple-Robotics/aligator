"""
Test function related to centroidal residuals
"""
import aligator
import numpy as np
from aligator import manifolds

np.random.seed(0)
FD_EPS = 1e-8
THRESH = 2 * FD_EPS**0.5

nx = 9
nk = 4
nu = 3 * nk
mass = 10.5
gravity = np.array([0, 0, -9.81])
space = manifolds.VectorSpace(9)
ndx = space.ndx


def sample_gauss(space):
    x0 = space.neutral()
    d = np.random.randn(space.ndx) * 0.1
    x1 = space.integrate(x0, d)
    return x0, d, x1


def test_com_translation():
    x0 = space.neutral()
    dx = np.random.randn(space.ndx) * 0.1
    x0 = space.integrate(x0, dx)
    u0 = np.zeros(nu)

    p_ref = np.array([0, 0, 0])
    fun = aligator.CentroidalCoMResidual(ndx, nu, p_ref)
    assert np.allclose(p_ref, fun.getReference())

    fdata = fun.createData()
    fun.evaluate(x0, fdata)

    assert np.allclose(fdata.value, x0[:3] - p_ref)

    fun_fd = aligator.FiniteDifferenceHelper(space, fun, FD_EPS)
    fdata2 = fun_fd.createData()
    fun_fd.evaluate(x0, u0, x0, fdata2)
    assert np.allclose(fdata.value, fdata2.value)

    fun_fd.computeJacobians(x0, u0, x0, fdata2)
    J_fd = fdata2.Jx
    assert fdata.Jx.shape == J_fd.shape

    for i in range(100):
        x, d, x0 = sample_gauss(space)
        fun.evaluate(x0, u0, x0, fdata)
        fun.computeJacobians(x0, u0, x0, fdata)
        fun_fd.evaluate(x0, u0, x0, fdata2)
        fun_fd.computeJacobians(x0, u0, x0, fdata2)
        assert np.allclose(fdata.Jx, fdata2.Jx, THRESH)


def test_linear_momentum():
    x0 = space.neutral()
    dx = np.random.randn(space.ndx) * 0.1
    x0 = space.integrate(x0, dx)
    u0 = np.zeros(nu)

    h_ref = np.array([0, 0, 0])
    fun = aligator.LinearMomentumResidual(ndx, nu, h_ref)
    assert np.allclose(h_ref, fun.getReference())

    fdata = fun.createData()
    fun.evaluate(x0, fdata)

    assert np.allclose(fdata.value, x0[3:6] - h_ref)

    fun_fd = aligator.FiniteDifferenceHelper(space, fun, FD_EPS)
    fdata2 = fun_fd.createData()
    fun_fd.evaluate(x0, u0, x0, fdata2)
    assert np.allclose(fdata.value, fdata2.value)

    fun_fd.computeJacobians(x0, u0, x0, fdata2)
    J_fd = fdata2.Jx
    assert fdata.Jx.shape == J_fd.shape

    for i in range(100):
        x, d, x0 = sample_gauss(space)
        fun.evaluate(x0, u0, x0, fdata)
        fun.computeJacobians(x0, u0, x0, fdata)
        fun_fd.evaluate(x0, u0, x0, fdata2)
        fun_fd.computeJacobians(x0, u0, x0, fdata2)
        assert np.allclose(fdata.Jx, fdata2.Jx, THRESH)


def test_angular_momentum():
    x0 = space.neutral()
    dx = np.random.randn(space.ndx) * 0.1
    x0 = space.integrate(x0, dx)
    u0 = np.zeros(nu)

    L_ref = np.array([0, 0, 0])
    fun = aligator.AngularMomentumResidual(ndx, nu, L_ref)
    assert np.allclose(L_ref, fun.getReference())

    fdata = fun.createData()
    fun.evaluate(x0, fdata)

    assert np.allclose(fdata.value, x0[6:] - L_ref)

    fun_fd = aligator.FiniteDifferenceHelper(space, fun, FD_EPS)
    fdata2 = fun_fd.createData()
    fun_fd.evaluate(x0, u0, x0, fdata2)
    assert np.allclose(fdata.value, fdata2.value)

    fun_fd.computeJacobians(x0, u0, x0, fdata2)
    J_fd = fdata2.Jx
    assert fdata.Jx.shape == J_fd.shape

    for i in range(100):
        x, d, x0 = sample_gauss(space)
        fun.evaluate(x0, u0, x0, fdata)
        fun.computeJacobians(x0, u0, x0, fdata)
        fun_fd.evaluate(x0, u0, x0, fdata2)
        fun_fd.computeJacobians(x0, u0, x0, fdata2)
        assert np.allclose(fdata.Jx, fdata2.Jx, THRESH)


def test_acceleration():
    x0 = space.neutral()
    dx = np.random.randn(space.ndx) * 0.1
    x0 = space.integrate(x0, dx)
    u0 = np.random.randn(nu)
    contact_map = [
        (True, np.array([0.2, 0.1, 0.0])),
        (True, np.array([0.2, 0.0, 0.0])),
        (True, np.array([0.0, 0.1, 0.0])),
        (True, np.array([0.0, 0.0, 0])),
    ]

    fun = aligator.CentroidalAccelerationResidual(ndx, nu, mass, gravity, contact_map)

    fdata = fun.createData()
    fun.evaluate(x0, u0, x0, fdata)

    comddot = np.zeros(3)
    for i in range(nk):
        comddot += u0[i * 3 : (i + 1) * 3]

    comddot /= mass
    comddot += gravity

    assert np.allclose(fdata.value, comddot)

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
        du = np.random.randn(nu) * 0.1
        u1 = u0 + du
        fun.evaluate(x0, u1, x0, fdata)
        fun.computeJacobians(x0, u1, x0, fdata)
        fun_fd.evaluate(x0, u1, x0, fdata2)
        fun_fd.computeJacobians(x0, u1, x0, fdata2)
        assert np.allclose(fdata.Ju, fdata2.Ju, THRESH)


def test_friction_cone():
    x0 = space.neutral()
    dx = np.random.randn(space.ndx) * 0.1
    x0 = space.integrate(x0, dx)
    u0 = np.random.randn(nu)
    k = 2
    mu = 0.5

    fun = aligator.FrictionConeResidual(ndx, nu, k, mu)

    fdata = fun.createData()
    fun.evaluate(x0, u0, x0, fdata)

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
        du = np.random.randn(nu) * 0.1
        u1 = u0 + du
        fun.evaluate(x0, u1, x0, fdata)
        fun.computeJacobians(x0, u1, x0, fdata)
        fun_fd.evaluate(x0, u1, x0, fdata2)
        fun_fd.computeJacobians(x0, u1, x0, fdata2)
        assert np.allclose(fdata.Ju, fdata2.Ju, THRESH)


def test_angular_acceleration():
    x0 = space.neutral()
    dx = np.random.randn(space.ndx) * 0.1
    x0 = space.integrate(x0, dx)
    u0 = np.random.randn(nu)
    contact_map = [
        (True, np.array([0.2, 0.1, 0.0])),
        (True, np.array([0.2, 0.0, 0.0])),
        (True, np.array([0.0, 0.1, 0.0])),
        (True, np.array([0.0, 0.0, 0])),
    ]
    fun = aligator.AngularAccelerationResidual(ndx, nu, mass, gravity, contact_map)

    fdata = fun.createData()
    fun.evaluate(x0, u0, x0, fdata)

    Ldot = np.zeros(3)
    for i in range(nk):
        Ldot += np.cross(fun.contact_map[i][1] - x0[:3], u0[i * 3 : (i + 1) * 3])

    assert np.allclose(fdata.value, Ldot)

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
        du = np.random.randn(nu) * 0.1
        dx = np.random.randn(space.ndx) * 0.1
        x1 = space.integrate(x0, dx)
        u1 = u0 + du
        fun.evaluate(x1, u1, x0, fdata)
        fun.computeJacobians(x1, u1, x0, fdata)
        fun_fd.evaluate(x1, u1, x0, fdata2)
        fun_fd.computeJacobians(x1, u1, x0, fdata2)
        assert np.allclose(fdata.Ju, fdata2.Ju, THRESH)
        assert np.allclose(fdata.Jx, fdata2.Jx, THRESH)


def test_wrapper_angular_acceleration():
    wx0 = space.neutral()
    wdx = np.random.randn(space.ndx) * 0.1
    wx0 = space.integrate(wx0, wdx)
    wu0 = np.random.randn(nu)

    ndx_w = 9 + nk * 3
    x0 = np.concatenate((wx0, wu0))
    u0 = np.random.randn(nu)
    wrapping_space = manifolds.VectorSpace(ndx_w)

    contact_map = [
        (True, np.array([0.2, 0.1, 0.0])),
        (True, np.array([0.2, 0.0, 0.0])),
        (True, np.array([0.0, 0.1, 0.0])),
        (True, np.array([0.0, 0.0, 0])),
    ]
    wrapped_fun = aligator.AngularAccelerationResidual(
        ndx, nu, mass, gravity, contact_map
    )
    fun = aligator.CentroidalWrapperResidual(wrapped_fun)

    fdata = fun.createData()
    wrapped_fdata = wrapped_fun.createData()
    wrapped_fun.evaluate(wx0, wu0, wx0, wrapped_fdata)
    fun.evaluate(x0, fdata)

    assert np.allclose(fdata.value, wrapped_fdata.value)

    fun_fd = aligator.FiniteDifferenceHelper(wrapping_space, fun, FD_EPS)
    fdata2 = fun_fd.createData()
    fun_fd.evaluate(x0, u0, x0, fdata2)
    assert np.allclose(fdata.value, fdata2.value)

    fun_fd.computeJacobians(x0, u0, x0, fdata2)
    J_fd = fdata2.Jx
    J_fd_u = fdata2.Ju
    assert fdata.Jx.shape == J_fd.shape
    assert fdata.Ju.shape == J_fd_u.shape

    for i in range(100):
        dx = np.random.randn(wrapping_space.ndx) * 0.1
        x1 = wrapping_space.integrate(x0, dx)
        fun.evaluate(x1, fdata)
        fun.computeJacobians(x1, fdata)
        fun_fd.evaluate(x1, u0, x1, fdata2)
        fun_fd.computeJacobians(x1, u0, x1, fdata2)
        assert np.allclose(fdata.Jx, fdata2.Jx, THRESH)


def test_wrapper_linear_momentum():
    wx0 = space.neutral()
    wdx = np.random.randn(space.ndx) * 0.1
    wx0 = space.integrate(wx0, wdx)
    wu0 = np.random.randn(nu)

    ndx_w = 9 + nk * 3
    x0 = np.concatenate((wx0, wu0))
    u0 = np.random.randn(nu)
    wrapping_space = manifolds.VectorSpace(ndx_w)

    h_ref = np.array([0, 0, 0])
    wrapped_fun = aligator.LinearMomentumResidual(ndx, nu, h_ref)
    fun = aligator.CentroidalWrapperResidual(wrapped_fun)

    fdata = fun.createData()
    wrapped_fdata = wrapped_fun.createData()
    wrapped_fun.evaluate(wx0, wu0, wx0, wrapped_fdata)
    fun.evaluate(x0, fdata)

    assert np.allclose(fdata.value, wrapped_fdata.value)

    fun_fd = aligator.FiniteDifferenceHelper(wrapping_space, fun, FD_EPS)
    fdata2 = fun_fd.createData()
    fun_fd.evaluate(x0, u0, x0, fdata2)
    assert np.allclose(fdata.value, fdata2.value)

    fun_fd.computeJacobians(x0, u0, x0, fdata2)
    J_fd = fdata2.Jx
    J_fd_u = fdata2.Ju
    assert fdata.Jx.shape == J_fd.shape
    assert fdata.Ju.shape == J_fd_u.shape

    for i in range(100):
        dx = np.random.randn(wrapping_space.ndx) * 0.1
        x1 = wrapping_space.integrate(x0, dx)
        fun.evaluate(x1, fdata)
        fun.computeJacobians(x1, fdata)
        fun_fd.evaluate(x1, u0, x0, fdata2)
        fun_fd.computeJacobians(x1, u0, x0, fdata2)
        assert np.allclose(fdata.Jx, fdata2.Jx, THRESH)


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main(sys.argv))
