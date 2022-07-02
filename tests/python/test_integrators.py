import numpy as np
from proxddp import dynamics, manifolds
import pytest


def finite_difference_explicit_dyn(
    dyn: dynamics.ExplicitIntegratorAbstract, x0, u0, eps
):
    data = dyn.createData()
    space: manifolds.ManifoldAbstract = dyn.space
    ode = dyn.differential_dynamics
    Jx_nd = np.zeros((dyn.ndx2, dyn.ndx1))
    ei = np.zeros(dyn.ndx1)
    dyn.forward(x0, u0, data)
    y0 = data.xout.copy()
    print("y0:   ", y0)
    yplus = y0.copy()
    for i in range(dyn.ndx1):
        ei[i] = eps
        xplus = space.integrate(x0, ei)
        dyn.forward(xplus, u0, data)
        yplus[:] = data.xout
        space.JintegrateTransport(x0, ei, yplus, 1)
        Jx_nd[:, i] = space.difference(y0, yplus) / eps
        ei[i] = 0.0

    Ju_nd = np.zeros((dyn.ndx2, dyn.nu))
    ei = np.zeros(dyn.nu)
    for i in range(dyn.nu):
        ei[i] = eps
        print("ei:", ei)
        uplus = u0 + eps
        dyn.forward(x0, uplus, data)
        yplus[:] = data.xout
        print("yplus:", yplus)
        print("dy:", yplus - y0)
        Ju_nd[:, i] = space.difference(y0, yplus) / eps
        ei[i] = 0.0
    return Jx_nd, Ju_nd


def create_multibody_ode():
    import pinocchio as pin

    model = pin.buildSampleModelHumanoid()
    space = manifolds.MultibodyPhaseSpace(model)
    nu = model.nv
    B = np.eye(nu)
    ode = dynamics.MultibodyFreeFwdDynamics(space, B)
    return ode


def create_linear():
    nx = 3
    nu = 2
    A = np.zeros((nx, nx))
    n = min(nx, nu)
    # A[1, 0] = 0.1
    B = np.zeros((nx, nu))
    B[range(n), range(n)] = 1.0
    # B[0, 0] = 0.5
    c = np.zeros(nx)
    ode = dynamics.LinearODE(A, B, c)
    cd = ode.createData()
    assert np.allclose(ode.A, cd.Jx)
    assert np.allclose(ode.B, cd.Ju)
    return ode


def test_explicit_euler():
    # ode = create_multibody_ode()
    ode = create_linear()
    dt = 0.1
    dyn = dynamics.IntegratorEuler(ode, dt)
    assert isinstance(dyn.createData(), dynamics.ExplicitIntegratorData)
    ode_int_run(ode, dyn)


def test_semi_euler():
    # ode = create_multibody_ode()
    ode = create_linear()
    dt = 0.1
    dyn = dynamics.IntegratorSemiImplEuler(ode, dt)
    assert isinstance(dyn.createData(), dynamics.ExplicitIntegratorData)
    ode_int_run(ode, dyn)


def ode_int_run(ode, dyn):
    x = ode.space.rand()
    u = np.random.randn(ode.nu)
    eps = 1e-4
    Jx_nd, Ju_nd = finite_difference_explicit_dyn(dyn, x, u, eps=eps)

    np.set_printoptions(precision=3, linewidth=250)
    data = dyn.createData()
    print(data.continuous_data)
    dyn.forward(x, u, data)
    dyn.dForward(x, u, data)
    err_x = abs(data.Jx - Jx_nd)
    err_u = abs(data.Ju - Ju_nd)

    print("err_x: {}".format(np.max(err_x)))
    print("err_u: {}".format(np.max(err_u)))
    print("err_x\n{}".format(err_x))
    print("err_u\n{}".format(err_u))
    print("Jx")
    print(data.Jx)
    print("jxnd")
    print(Jx_nd)
    print("Ju")
    print(data.Ju)
    print(Ju_nd)
    assert np.allclose(data.Jx, Jx_nd)
    assert np.allclose(data.Ju, Ju_nd)


if __name__ == "__main__":
    import sys

    retcode = pytest.main(sys.argv)
