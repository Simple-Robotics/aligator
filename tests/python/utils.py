import numpy as np
import aligator

from aligator import manifolds, dynamics


def infNorm(x):
    return np.linalg.norm(x, np.inf)


def create_multibody_ode(humanoid=True):
    try:
        import pinocchio as pin

        if humanoid:
            model = pin.buildSampleModelHumanoid()
        else:
            model = pin.buildSampleModelManipulator()
        space = manifolds.MultibodyPhaseSpace(model)
        nu = model.nv
        B = np.eye(nu)
        ode = dynamics.MultibodyFreeFwdDynamics(space, B)
        data = ode.createData()
        assert isinstance(data, dynamics.MultibodyFreeFwdData)
        return ode
    except ImportError:
        return None


def create_linear_ode(nx, nu):
    n = min(nx, nu)
    A = np.eye(nx)
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


def ode_finite_difference(dyn: dynamics.ODEAbstract, space, x, u, eps=1e-8):
    assert isinstance(dyn, dynamics.ODEAbstract)
    ndx = space.ndx
    Jx = np.zeros((ndx, ndx))
    dx = np.zeros(ndx)
    data = dyn.createData()
    dyn.forward(x, u, data)
    # distance to origin
    _dx = space.difference(space.neutral(), x)
    ex = eps * max(1.0, np.linalg.norm(_dx))
    f = data.xdot.copy()
    for i in range(ndx):
        dx[i] = ex
        dyn.forward(space.integrate(x, dx), u, data)
        fp = data.xdot.copy()
        dyn.forward(space.integrate(x, -dx), u, data)
        fm = data.xdot.copy()
        Jx[:, i] = (fp - fm) / (2 * ex)
        dx[i] = 0.0

    nu = u.shape[0]
    eu = eps * max(1.0, np.linalg.norm(u))
    Ju = np.zeros((ndx, nu))
    du = np.zeros(nu)
    data = dyn.createData()
    for i in range(nu):
        du[i] = eu
        dyn.forward(x, u + du, data)
        fp[:] = data.xdot
        Ju[:, i] = (fp - f) / eu
        du[i] = 0.0

    return Jx, Ju


def cost_finite_grad(costmodel, space, x, u, eps=1e-8):
    ndx = space.ndx
    nu = u.size
    grad = np.zeros(ndx + nu)
    dx = np.zeros(ndx)
    du = np.zeros(nu)
    data = costmodel.createData()
    costmodel.evaluate(x, u, data)
    # distance to origin
    _dx = space.difference(space.neutral(), x)
    ex = eps * max(1.0, np.linalg.norm(_dx))
    vref = data.value
    for i in range(ndx):
        dx[i] = ex
        x1 = space.integrate(x, dx)
        costmodel.evaluate(x1, u, data)
        grad[i] = (data.value - vref) / ex
        dx[i] = 0.0

    for i in range(ndx, ndx + nu):
        du[i - ndx] = ex
        u1 = u + du
        costmodel.evaluate(x, u1, data)
        grad[i] = (data.value - vref) / ex
        du[i - ndx] = 0.0

    return grad


def function_finite_difference(
    fun: aligator.StageFunction,
    space: manifolds.ManifoldAbstract,
    x0,
    u0,
    eps=1e-8,
):
    """Use finite differences to compute Jacobians
    of a `aligator.StageFunction`.
    """
    data = fun.createData()
    Jx_nd = np.zeros((fun.nr, fun.ndx1))
    ei = np.zeros(fun.ndx1)
    fun.evaluate(x0, u0, data)
    r0 = data.value.copy()
    for i in range(fun.ndx1):
        ei[i] = eps
        xplus = space.integrate(x0, ei)
        fun.evaluate(xplus, u0, data)
        Jx_nd[:, i] = (data.value - r0) / eps
        ei[i] = 0.0

    ei = np.zeros(fun.nu)
    Ju_nd = np.zeros((fun.nr, fun.nu))
    for i in range(fun.nu):
        ei[i] = eps
        fun.evaluate(x0, u0 + ei, data)
        Ju_nd[:, i] = (data.value - r0) / eps
        ei[i] = 0.0

    return Jx_nd, Ju_nd
