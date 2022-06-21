import proxddp
import numpy as np


def test_custom_controlbox():
    nx = 3
    nu = 3

    class ControlBoxFunction(proxddp.StageFunction):
        def __init__(self, nx, nu, u_min, u_max) -> None:
            super().__init__(nx, nu, 2 * nu)
            self.u_min = u_min
            self.u_max = u_max

        def evaluate(self, x, u, y, data):
            data.value[:] = np.concatenate([self.u_min - u, u - self.u_max])

        def computeJacobians(self, x, u, y, data):
            nu = u.shape[0]
            data.Ju[:, :] = np.block([[-np.eye(nu)], [np.eye(nu)]])

    u_min = np.ones(nu) * -0.1
    u_max = np.ones(nu) * 0.1
    box_function = ControlBoxFunction(nx, nu, u_min, u_max)
    bf2 = proxddp.ControlBoxFunction(nx, u_min, u_max)
    data1 = box_function.createData()
    data2 = bf2.createData()

    x0 = np.random.randn(nx)
    u0 = np.random.randn(nu)
    print(u0)

    box_function.evaluate(x0, u0, x0, data1)
    box_function.computeJacobians(x0, u0, x0, data1)
    bf2.evaluate(x0, u0, x0, data2)
    bf2.computeJacobians(x0, u0, x0, data2)
    print(data1.value)
    print(data1.Ju)
    print("bf2 d:", bf2.d)
    assert np.allclose(data1.value, data2.value)
    assert np.allclose(data1.jac_buffer_, data2.jac_buffer_)
