import numpy as np

from aligator import manifolds, dynamics


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
    f = data.xdot.copy()
    fp = f.copy()
    for i in range(ndx):
        dx[i] = EPS
        x_p = space.integrate(x, dx)
        dynmodel.forward(x_p, u, data)
        fp[:] = data.xdot
        Jx[:, i] = (fp - f) / EPS
        dx[i] = 0.0

    nu = u.shape[0]
    Ju = np.zeros((ndx, nu))
    du = np.zeros(nu)
    data = dynmodel.createData()
    for i in range(nu):
        du[i] = EPS
        dynmodel.forward(x, u + du, data)
        fp[:] = data.xdot
        Ju[:, i] = (fp - f) / EPS
        du[i] = 0.0

    return Jx, Ju
