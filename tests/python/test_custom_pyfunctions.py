import proxddp
import numpy as np
import pytest

from proxddp import ControlBoxFunction as PyControlBoxFunction


def test_abstract():
    space = proxddp.manifolds.SE2()
    ndx = space.ndx
    nu = 3
    nr = 1
    fun = proxddp.StageFunction(ndx, nu, nr)
    data = fun.createData()
    print(data)


def test_custom_controlbox():
    space = proxddp.manifolds.SE2()
    ndx = space.ndx
    nu = 3

    u_min = np.ones(nu) * -0.1
    u_max = np.ones(nu) * 0.1
    box_function = PyControlBoxFunction(ndx, u_min, u_max)
    bf2 = proxddp.ControlBoxFunction(ndx, u_min, u_max)
    data1: proxddp.StageFunctionData = box_function.createData()
    data2 = bf2.createData()

    lbd0 = np.zeros(box_function.nr)
    x0 = np.random.randn(ndx)
    u0 = np.random.randn(nu)

    box_function.evaluate(x0, u0, x0, data1)
    box_function.computeJacobians(x0, u0, x0, data1)
    bf2.evaluate(x0, u0, x0, data2)
    bf2.computeJacobians(x0, u0, x0, data2)
    print(data1.value)
    print(data1.Ju)
    assert np.allclose(data1.value, data2.value)
    assert np.allclose(data1.jac_buffer, data2.jac_buffer)

    # expected behavior: initial value of vhp_buffer is 0
    assert np.allclose(data1.vhp_buffer, 0.0)

    rdm = np.random.randn(*data1.vhp_buffer.shape)
    data1.vhp_buffer[:, :] = rdm
    box_function.computeVectorHessianProducts(x0, u0, x0, lbd0, data1)
    # expected behavior: unimplemented computeVectorHessianProducts does nothing.
    assert np.allclose(data1.vhp_buffer, rdm)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
