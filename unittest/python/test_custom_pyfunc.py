import proxddp
import numpy as np


def test_custom_controlbox():
    nx = 3
    nu = 3

    class ControlBoxFunction(proxddp.StageFunction):
        def __init__(self, nx, nu, nr, u_min, u_max) -> None:
            super().__init__(nx, nu, nr)
            self.u_min = u_min
            self.u_max = u_max

        def evaluate(self, x, u, y, data):
            data.value[:] = np.concatenate([u - self.u_max, self.u_min - u])

        def computeJacobians(self, x, u, y, data):
            nu = u.shape[0]
            data.Ju[:, :] = np.block([[np.eye(nu)], [-np.eye(nu)]])

    u_min = np.ones(nu) * -0.1
    u_max = np.ones(nu) * 0.1
    box_function = ControlBoxFunction(nx, nu, 2 * nu, u_min, u_max)
    data = box_function.createData()

    x0 = np.random.randn(nx)
    u0 = np.random.randn(nu)
    print(u0)

    box_function.evaluate(x0, u0, x0, data)
    box_function.computeJacobians(x0, u0, x0, data)
    print(data.value)
    print(data.Ju)
