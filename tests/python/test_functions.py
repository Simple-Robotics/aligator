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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
