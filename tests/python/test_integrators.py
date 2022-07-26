import numpy as np
from proxddp import dynamics, manifolds
import pytest


def finite_difference_explicit_dyn(
    dyn: dynamics.ExplicitIntegratorAbstract, x0, u0, eps
):
    data = dyn.createData()
    space: manifolds.ManifoldAbstract = dyn.space
    Jx_nd = np.zeros((dyn.ndx2, dyn.ndx1))
    ei = np.zeros(dyn.ndx1)
    dyn.forward(x0, u0, data)
    y0 = data.xout.copy()
    yplus = y0.copy()
    for i in range(dyn.ndx1):
        ei[i] = eps
        xplus = space.integrate(x0, ei)
        dyn.forward(xplus, u0, data)
        yplus[:] = data.xout
        space.JintegrateTransport(x0, ei, yplus, 1)
        Jx_nd[:, i] = space.difference(y0, yplus) / eps
        ei[i] = 0.0

    uspace = manifolds.VectorSpace(dyn.nu)
    Ju_nd = np.zeros((dyn.ndx2, dyn.nu))
    ei = np.zeros(dyn.nu)
    for i in range(dyn.nu):
        ei[i] = eps
        uplus = uspace.integrate(u0, ei)
        dyn.forward(x0, uplus, data)
        yplus[:] = data.xout
        Ju_nd[:, i] = space.difference(y0, yplus) / eps
        ei[:] = 0.0
    return Jx_nd, Ju_nd


def create_multibody_ode():
    import pinocchio as pin

    model = pin.buildSampleModelHumanoid()
    space = manifolds.MultibodyPhaseSpace(model)
    nu = model.nv
    B = np.eye(nu)
    ode = dynamics.MultibodyFreeFwdDynamics(space, B)
    return ode


def create_linear_ode(nx, nu):
    A = np.zeros((nx, nx))
    n = min(nx, nu)
    A[1, 0] = 0.1
    B = np.zeros((nx, nu))
    B[range(n), range(n)] = 1.0
    B[0, 1] = 0.5
    c = np.zeros(nx)
    ode = dynamics.LinearODE(A, B, c)
    cd = ode.createData()
    assert np.allclose(ode.A, cd.Jx)
    assert np.allclose(ode.B, cd.Ju)
    return ode


@pytest.mark.parametrize("ode", [create_linear_ode(4, 2), create_multibody_ode()])
@pytest.mark.parametrize(
    "integrator",
    [
        dynamics.IntegratorEuler,
        dynamics.IntegratorSemiImplEuler,
        dynamics.IntegratorRK2,
    ],
)
def test_ode_int_combinations(ode, integrator):
    dt = 0.1
    dyn = integrator(ode, dt)
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
    assert np.allclose(data.Jx, Jx_nd)
    assert np.allclose(data.Ju, Ju_nd)


if __name__ == "__main__":
    import sys

    retcode = pytest.main(sys.argv)
