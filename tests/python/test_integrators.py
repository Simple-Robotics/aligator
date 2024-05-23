import numpy as np
import aligator
from aligator import dynamics, manifolds
import pytest
from utils import create_linear_ode, create_multibody_ode

EPSILON = 1e-5
ATOL = EPSILON**0.5


def function_finite_difference(
    fun: aligator.StageFunction,
    space: manifolds.ManifoldAbstract,
    x0,
    u0,
    y0=None,
    eps=EPSILON,
):
    """Use finite differences to compute Jacobians
    of a `aligator.StageFunction`.

    TODO: move to a test utils file
    """
    if y0 is None:
        y0 = x0
    data = fun.createData()
    Jx_nd = np.zeros((fun.nr, fun.ndx1))
    ei = np.zeros(fun.ndx1)
    fun.evaluate(x0, u0, y0, data)
    r0 = data.value.copy()
    for i in range(fun.ndx1):
        ei[i] = eps
        xplus = space.integrate(x0, ei)
        fun.evaluate(xplus, u0, y0, data)
        Jx_nd[:, i] = (data.value - r0) / eps
        ei[i] = 0.0

    ei = np.zeros(fun.nu)
    Ju_nd = np.zeros((fun.nr, fun.nu))
    for i in range(fun.nu):
        ei[i] = eps
        fun.evaluate(x0, u0 + ei, y0, data)
        Ju_nd[:, i] = (data.value - r0) / eps
        ei[i] = 0.0

    ei = np.zeros(fun.ndx2)
    yplus = y0.copy()
    Jy_nd = np.zeros((fun.nr, fun.ndx2))
    for i in range(fun.ndx2):
        ei[i] = eps
        space.integrate(y0, ei, yplus)
        fun.evaluate(x0, u0, yplus, data)
        Jy_nd[:, i] = (data.value - r0) / eps
        ei[i] = 0.0

    return Jx_nd, Ju_nd, Jy_nd


def finite_difference_explicit_dyn(dyn: dynamics.IntegratorAbstract, x0, u0, eps):
    data = dyn.createData()
    space: manifolds.ManifoldAbstract = dyn.space
    Jx_nd = np.zeros((dyn.ndx2, dyn.ndx1))
    ei = np.zeros(dyn.ndx1)
    dyn.forward(x0, u0, data)
    y0 = data.xnext.copy()
    yplus = y0.copy()
    for i in range(dyn.ndx1):
        ei[i] = eps
        xplus = space.integrate(x0, ei)
        dyn.forward(xplus, u0, data)
        yplus[:] = data.xnext
        Jx_nd[:, i] = space.difference(y0, yplus) / eps
        ei[i] = 0.0

    uspace = manifolds.VectorSpace(dyn.nu)
    Ju_nd = np.zeros((dyn.ndx2, dyn.nu))
    ei = np.zeros(dyn.nu)
    for i in range(dyn.nu):
        ei[i] = eps
        uplus = uspace.integrate(u0, ei)
        dyn.forward(x0, uplus, data)
        yplus[:] = data.xnext
        Ju_nd[:, i] = space.difference(y0, yplus) / eps
        ei[:] = 0.0
    return Jx_nd, Ju_nd


@pytest.mark.parametrize("ode", [create_linear_ode(4, 2), create_multibody_ode()])
@pytest.mark.parametrize(
    "integrator",
    [
        dynamics.IntegratorEuler,
        dynamics.IntegratorSemiImplEuler,
        dynamics.IntegratorRK2,
    ],
)
def test_explicit_integrator_combinations(ode, integrator):
    dt = 0.1
    if ode is None:
        return True
    dyn = integrator(ode, dt)
    ode_int_run(ode, dyn)


@pytest.mark.parametrize("dae", [create_linear_ode(4, 3), create_multibody_ode()])
@pytest.mark.parametrize("integrator", [dynamics.IntegratorMidpoint])
def test_implicit_integrator(
    dae: dynamics.ContinuousDynamicsAbstract, integrator: dynamics.IntegratorAbstract
):
    dt = 0.1
    dyn = integrator(dae, dt)
    x = dae.space.rand()
    x = np.clip(x, -5, 5)
    u = np.random.randn(dyn.nu)
    data = dyn.createData()
    dyn.evaluate(x, u, x, data)
    assert isinstance(data, dynamics.IntegratorData)

    Jx_nd, Ju_nd, Jy_nd = function_finite_difference(dyn, dyn.space, x, u)

    dyn.evaluate(x, u, x, data)
    dyn.computeJacobians(x, u, x, data)
    assert np.allclose(data.Jx, Jx_nd, atol=ATOL)
    assert np.allclose(data.Ju, Ju_nd, atol=ATOL)
    assert np.allclose(data.Jy, Jy_nd, atol=ATOL)


def exp_dyn_fd_check(dyn, x, u, eps=EPSILON):
    Jx_nd, Ju_nd = finite_difference_explicit_dyn(dyn, x, u, eps=eps)

    np.set_printoptions(precision=3, linewidth=250)
    data = dyn.createData()
    dyn.forward(x, u, data)
    dyn.dForward(x, u, data)
    assert np.allclose(data.Jx, Jx_nd, atol=ATOL)
    assert np.allclose(data.Ju, Ju_nd, atol=ATOL)


def ode_int_run(ode, dyn):
    x = ode.space.rand()
    x = np.clip(x, -5, 5)
    u = np.random.randn(ode.nu)

    exp_dyn_fd_check(dyn, x, u)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
