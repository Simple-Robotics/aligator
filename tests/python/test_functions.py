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
    assert np.allclose(data3.jac_buffer, A @ sd3.jac_buffer)


def test_slicing():
    space = manifolds.R4()
    nr = 7
    nx = space.nx
    A = np.random.randn(nr, nx)
    nu = 2
    B = np.random.randn(nr, nu)
    c = np.zeros(nr)
    fn = proxddp.LinearFunction(A, B, c)
    x0 = space.rand()
    u0 = np.zeros(fn.nu)

    dfull = fn.createData()

    idxs = [0, 2]
    fslice = fn[idxs]
    dslice = fslice.createData()
    assert idxs == fslice.indices.tolist()

    fn.evaluate(x0, u0, x0, dfull)
    fslice.evaluate(x0, u0, x0, dslice)

    assert np.allclose(dfull.value[idxs], dslice.value)

    fn.computeJacobians(x0, u0, x0, dfull)
    fslice.computeJacobians(x0, u0, x0, dslice)

    assert np.allclose(dfull.jac_buffer[idxs, :], dslice.jac_buffer)

    fs2 = fn[1::2]
    print("slice size:", fs2.nr)
    idx2 = [1, 3, 5]
    assert fs2.indices.tolist() == idx2

    fs3 = fn[1]
    assert fs3.indices.tolist() == [1]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
