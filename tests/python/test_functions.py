import proxddp
import numpy as np
import pytest

from proxddp import manifolds


def test_manifold_diff():
    space = manifolds.SE2()
    nu = 2
    target = space.rand()
    fun = proxddp.StateErrorResidual(space, nu, target)

    data = fun.createData()

    x = space.rand()
    u = np.random.randn(nu)
    y = x
    fun.evaluate(x, u, y, data)
    assert np.allclose(data.value, space.difference(target, x))
    cp = data.value.copy()
    cp[0] = -1.0
    data.value[0] = -1.0
    assert np.allclose(data.value, cp)

    fun.computeJacobians(x, u, y, data)

    # TEST SLICING

    idx = [1, 2]
    fun_slice = proxddp.FunctionSliceXpr(fun, idx)
    assert fun_slice.nr == 2
    data2 = fun_slice.createData()
    print("DATA 2 OK:", data2)
    fun_slice.evaluate(x, u, y, data2)
    assert np.allclose(data2.value[: len(idx)], data.value[idx])

    fun_slice.computeJacobians(x, u, y, data2)
    assert np.allclose(data2.jac_buffer_[: len(idx)], data.jac_buffer_[idx])

    # TEST LINEAR COMPOSE
    A = np.array([[1.0, 1.0, 0.0]])
    b = np.array([0.0])
    assert A.shape[1] == fun.nr
    fun_lin = proxddp.LinearFunctionComposition(fun, A, b)
    assert fun_lin.nr == A.shape[0]
    data3 = fun_lin.createData()
    sd3 = data3.sub_data
    fun_lin.evaluate(x, u, y, data3)
    print("d3 value:", data3.value)
    print(sd3.value)
    assert np.allclose(data3.value, A @ sd3.value + b)

    fun_lin.computeJacobians(x, u, y, data3)
    assert np.allclose(data3.jac_buffer_, A @ sd3.jac_buffer_)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
