import numpy as np

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


def finite_diff(dynmodel, space, x, u, EPS=1e-8):
    ndx = space.ndx
    Jx = np.zeros((ndx, ndx))
    dx = np.zeros(ndx)
    data = dynmodel.createData()
    dynmodel.forward(x, u, data)
    # distance to origin
    _dx = space.difference(space.neutral(), x)
    ex = EPS * max(1.0, np.linalg.norm(_dx))
    f = data.xdot.copy()
    for i in range(ndx):
        dx[i] = ex
        dynmodel.forward(space.integrate(x, dx), u, data)
        fp = data.xdot.copy()
        dynmodel.forward(space.integrate(x, -dx), u, data)
        fm = data.xdot.copy()
        Jx[:, i] = (fp - fm) / (2 * ex)
        dx[i] = 0.0

    nu = u.shape[0]
    eu = EPS * max(1.0, np.linalg.norm(u))
    Ju = np.zeros((ndx, nu))
    du = np.zeros(nu)
    data = dynmodel.createData()
    for i in range(nu):
        du[i] = eu
        dynmodel.forward(x, u + du, data)
        fp[:] = data.xdot
        Ju[:, i] = (fp - f) / eu
        du[i] = 0.0

    return Jx, Ju


def cost_finite_grad(costmodel, space, x, u, EPS=1e-8):
    ndx = space.ndx
    nu = u.size
    grad = np.zeros(ndx + nu)
    dx = np.zeros(ndx)
    du = np.zeros(nu)
    data = costmodel.createData()
    costmodel.evaluate(x, u, data)
    # distance to origin
    _dx = space.difference(space.neutral(), x)
    ex = EPS * max(1.0, np.linalg.norm(_dx))
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
