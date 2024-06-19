import typing

import numpy as np

from aligator import (
    manifolds,
    dynamics,
    CommonModelContainer,
    CommonModelDataContainer,
    CommonModelBuilderContainer,
)


def infNorm(x):
    return np.linalg.norm(x, np.inf)


class CommonContainers(typing.NamedTuple):
    models: CommonModelContainer
    datas: CommonModelDataContainer

    def evaluate(self, x: np.ndarray, u: np.ndarray):
        self.models.evaluate(x, u, self.datas)

    def compute_gradients(self, x: np.ndarray, u: np.ndarray):
        self.models.computeGradients(x, u, self.datas)


def configure_functions(
    functions, common_builders: typing.Optional[CommonModelBuilderContainer] = None
) -> CommonContainers:
    if common_builders is None:
        common_builders = CommonModelBuilderContainer()
    for f in functions:
        f.configure(common_builders)
    common_models = common_builders.createCommonModelContainer()
    common_datas = common_models.createData()
    return CommonContainers(common_models, common_datas)


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
        return dynamics.MultibodyFreeFwdDynamics(space, B)
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
    return dynamics.LinearODE(A, B, c)


def finite_diff(dynmodel, space, x, u, EPS=1e-8):
    ndx = space.ndx
    Jx = np.zeros((ndx, ndx))
    dx = np.zeros(ndx)
    common_containers = configure_functions([dynmodel])
    data = dynmodel.createData(common_containers.datas)

    common_containers.evaluate(x, u)
    dynmodel.forward(x, u, data)
    # distance to origin
    _dx = space.difference(space.neutral(), x)
    ex = EPS * max(1.0, np.linalg.norm(_dx))
    f = data.xdot.copy()
    for i in range(ndx):
        dx[i] = ex
        x_plus = space.integrate(x, dx)
        common_containers.evaluate(x_plus, u)
        dynmodel.forward(x_plus, u, data)
        fp = data.xdot.copy()

        x_minus = space.integrate(x, -dx)
        common_containers.evaluate(x_minus, u)
        dynmodel.forward(x_minus, u, data)
        fm = data.xdot.copy()
        Jx[:, i] = (fp - fm) / (2 * ex)
        dx[i] = 0.0

    nu = u.shape[0]
    eu = EPS * max(1.0, np.linalg.norm(u))
    Ju = np.zeros((ndx, nu))
    du = np.zeros(nu)
    data = dynmodel.createData(common_containers.datas)
    for i in range(nu):
        du[i] = eu
        u_plus = u + du
        common_containers.evaluate(x, u_plus)
        dynmodel.forward(x, u_plus, data)
        fp[:] = data.xdot
        Ju[:, i] = (fp - f) / eu
        du[i] = 0.0

    return Jx, Ju
