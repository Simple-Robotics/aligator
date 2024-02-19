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
sample_factor = 0.1


def sample_gauss(space):
    x0 = space.neutral()
    d = np.random.randn(space.ndx) * sample_factor
    x1 = space.integrate(x0, d)
    return x0, d, x1


def test_contact_map():
    contact_states = [True, False, True]
    contact_poses = [
        np.array([0.2, 0.1, 0.0]),
        np.array([0.2, 0.0, 0.0]),
        np.array([0.0, 0.1, 0.0]),
    ]
    contact_map = aligator.ContactMap(contact_states, contact_poses)
    contact_map.addContact(False, np.array([0.0, 0.0, 0.0]))

    assert contact_map.size == 4

    contact_map.removeContact(3)

    assert contact_map.size == 3


def test_com_translation():
    x, d, x0 = sample_gauss(space)
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
    x, d, x0 = sample_gauss(space)
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
    x, d, x0 = sample_gauss(space)
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
    x, d, x0 = sample_gauss(space)
    force_size = 3
    nu = force_size * nk
    u0 = np.random.randn(nu)

    contact_states = [True, False, True, True]
    contact_poses = [
        np.array([0.2, 0.1, 0.0]),
        np.array([0.2, 0.0, 0.0]),
        np.array([0.0, 0.1, 0.0]),
        np.array([0.0, 0.0, 0]),
    ]
    contact_map = aligator.ContactMap(contact_states, contact_poses)

    fun = aligator.CentroidalAccelerationResidual(
        ndx, nu, mass, gravity, contact_map, force_size
    )
    fdata = fun.createData()
    fun.evaluate(x0, u0, x0, fdata)

    comddot = np.zeros(3)
    for i in range(nk):
        if fun.contact_map.contact_states[i]:
            comddot += u0[i * force_size : i * force_size + 3]

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

    force_size = 6
    nu = force_size * nk
    u0 = np.random.randn(nu)

    fun = aligator.CentroidalAccelerationResidual(
        ndx, nu, mass, gravity, contact_map, force_size
    )
    fdata = fun.createData()
    fun.evaluate(x0, u0, x0, fdata)

    comddot = np.zeros(3)
    for i in range(nk):
        if fun.contact_map.contact_states[i]:
            comddot += u0[i * force_size : i * force_size + 3]

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
    x, d, x0 = sample_gauss(space)
    nu = 3 * nk
    u0 = np.random.randn(nu)
    k = 2
    mu = 0.5
    epsilon = 1e-3

    fun = aligator.FrictionConeResidual(ndx, nu, k, mu, epsilon)

    fdata = fun.createData()
    fun.evaluate(x0, u0, x0, fdata)
    fcone = np.zeros(2)
    fcone[0] = epsilon - u0[k * 3 + 2]
    fcone[1] = -(mu**2) * u0[k * 3 + 2] ** 2 + u0[k * 3] ** 2 + u0[k * 3 + 1] ** 2

    assert np.allclose(fdata.value, fcone)

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
        du = np.random.randn(nu) * sample_factor
        u1 = u0 + du
        fun.evaluate(x0, u1, x0, fdata)
        fun.computeJacobians(x0, u1, x0, fdata)
        fun_fd.evaluate(x0, u1, x0, fdata2)
        fun_fd.computeJacobians(x0, u1, x0, fdata2)
        assert np.allclose(fdata.Ju, fdata2.Ju, THRESH)


def test_wrench_cone():
    x, d, x0 = sample_gauss(space)
    force_size = 6
    nu = force_size * nk
    u0 = np.random.randn(nu)
    k = 2
    mu = 0.5
    L = 0.1
    W = 0.05

    fun = aligator.WrenchConeResidual(ndx, nu, k, mu, L, W)
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
        du = np.random.randn(nu) * sample_factor
        u1 = u0 + du
        fun.evaluate(x0, u1, x0, fdata)
        fun.computeJacobians(x0, u1, x0, fdata)
        fun_fd.evaluate(x0, u1, x0, fdata2)
        fun_fd.computeJacobians(x0, u1, x0, fdata2)
        assert np.allclose(fdata.Ju, fdata2.Ju, THRESH)


def test_angular_acceleration():
    x, d, x0 = sample_gauss(space)
    force_size = 3
    nu = force_size * nk
    u0 = np.random.randn(nu)
    contact_states = [True, False, True, True]
    contact_poses = [
        np.array([0.2, 0.1, 0.0]),
        np.array([0.2, 0.0, 0.0]),
        np.array([0.0, 0.1, 0.0]),
        np.array([0.0, 0.0, 0]),
    ]
    contact_map = aligator.ContactMap(contact_states, contact_poses)

    fun = aligator.AngularAccelerationResidual(
        ndx, nu, mass, gravity, contact_map, force_size
    )
    fdata = fun.createData()

    fun.evaluate(x0, u0, x0, fdata)

    Ldot = np.zeros(3)
    for i in range(nk):
        if contact_states[i]:
            Ldot += np.cross(contact_poses[i] - x0[:3], u0[i * 3 : (i + 1) * 3])

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
        du = np.random.randn(nu) * sample_factor
        u1 = u0 + du
        x, d, x0 = sample_gauss(space)
        fun.evaluate(x0, u1, x0, fdata)
        fun.computeJacobians(x0, u1, x0, fdata)
        fun_fd.evaluate(x0, u1, x0, fdata2)
        fun_fd.computeJacobians(x0, u1, x0, fdata2)
        assert np.allclose(fdata.Ju, fdata2.Ju, THRESH)
        assert np.allclose(fdata.Jx, fdata2.Jx, THRESH)

    force_size = 6
    nu = force_size * nk
    u0 = np.random.randn(nu)

    fun = aligator.AngularAccelerationResidual(
        ndx, nu, mass, gravity, contact_map, force_size
    )
    fdata = fun.createData()

    fun.evaluate(x0, u0, x0, fdata)

    Ldot = np.zeros(3)
    for i in range(nk):
        if contact_states[i]:
            Ldot += np.cross(contact_poses[i] - x0[:3], u0[i * 6 : i * 6 + 3])
            Ldot += u0[i * 6 + 3 : (i + 1) * 6]

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
        du = np.random.randn(nu) * sample_factor
        u1 = u0 + du
        x, d, x0 = sample_gauss(space)
        fun.evaluate(x0, u1, x0, fdata)
        fun.computeJacobians(x0, u1, x0, fdata)
        fun_fd.evaluate(x0, u1, x0, fdata2)
        fun_fd.computeJacobians(x0, u1, x0, fdata2)
        assert np.allclose(fdata.Ju, fdata2.Ju, THRESH)
        assert np.allclose(fdata.Jx, fdata2.Jx, THRESH)


def test_wrapper_angular_acceleration():
    wx, d, wx0 = sample_gauss(space)
    wu0 = np.random.randn(nu)

    force_size = 3
    ndx_w = 9 + nk * force_size
    x0 = np.concatenate((wx0, wu0))
    u0 = np.random.randn(nu)
    wrapping_space = manifolds.VectorSpace(ndx_w)

    contact_states = [True, False, True, True]
    contact_poses = [
        np.array([0.2, 0.1, 0.0]),
        np.array([0.2, 0.0, 0.0]),
        np.array([0.0, 0.1, 0.0]),
        np.array([0.0, 0.0, 0]),
    ]
    contact_map = aligator.ContactMap(contact_states, contact_poses)

    wrapped_fun = aligator.AngularAccelerationResidual(
        ndx, nu, mass, gravity, contact_map, force_size
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
        x, d, x1 = sample_gauss(wrapping_space)
        fun.evaluate(x1, fdata)
        fun.computeJacobians(x1, fdata)
        fun_fd.evaluate(x1, u0, x1, fdata2)
        fun_fd.computeJacobians(x1, u0, x1, fdata2)
        assert np.allclose(fdata.Jx, fdata2.Jx, THRESH)


def test_wrapper_linear_momentum():
    wx, d, wx0 = sample_gauss(space)
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
        x, d, x1 = sample_gauss(wrapping_space)
        fun.evaluate(x1, fdata)
        fun.computeJacobians(x1, fdata)
        fun_fd.evaluate(x1, u0, x0, fdata2)
        fun_fd.computeJacobians(x1, u0, x0, fdata2)
        assert np.allclose(fdata.Jx, fdata2.Jx, THRESH)


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main(sys.argv))
