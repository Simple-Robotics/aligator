from proxddp import QuadraticCost, CostStack
from proxddp import manifolds
import proxddp
import numpy as np

import pytest


def test_cost_stack():
    nx = 2
    nu = 2
    cost_stack = CostStack(nx, nu)
    Q = np.random.randn(4, nx)
    Q = Q.T @ Q / nx
    R = np.eye(nu)
    rcost = QuadraticCost(Q, R)

    cost_stack.addCost(rcost, 1.0)
    data1 = rcost.createData()
    data2 = cost_stack.createData()

    for _ in range(10):
        x0 = np.random.randn(nx)
        u0 = np.random.randn(nu)

        rcost.evaluate(x0, u0, data1)
        cost_stack.evaluate(x0, u0, data2)
        rcost.computeGradients(x0, u0, data1)
        cost_stack.computeGradients(x0, u0, data2)

        rcost.computeHessians(x0, u0, data1)
        cost_stack.computeHessians(x0, u0, data2)

        assert data1.value == data2.value
        assert np.allclose(data1.grad, data2.grad)
        assert np.allclose(data1.hess, data2.hess)

        assert np.allclose(data1.Lxx, Q)
        assert np.allclose(data1.Luu, R)


def test_composite_cost():
    space = manifolds.VectorSpace(3)
    ndx = space.ndx
    x0 = space.rand()

    nu = space.ndx
    u0 = np.ones(nu)
    target = space.rand()
    fun = proxddp.StateErrorResidual(space, nu, target)
    # for debug
    fd = fun.createData()
    fun.evaluate(x0, u0, x0, fd)
    fun.computeJacobians(x0, u0, x0, fd)

    # costs

    np.random.seed(40)

    weights = np.random.randn(4, fun.nr)
    weights = weights.T @ weights
    cost = proxddp.QuadraticResidualCost(fun, weights)
    assert np.array_equal(weights, cost.weights)

    data = cost.createData()
    print("Composite data:", data)
    assert isinstance(data, proxddp.CompositeCostData)

    cost.evaluate(x0, u0, data)
    cost.computeGradients(x0, u0, data)
    cost.computeHessians(x0, u0, data)

    J = fd.jac_buffer[:, : ndx + nu]
    ref_grad = J.T @ weights @ fd.value
    ref_hess = J.T @ weights @ J

    print("QuadCost:")
    print(data.value)
    print(data.grad)
    print(data.hess)
    assert np.allclose(data.grad, ref_grad)
    assert np.allclose(data.hess, ref_hess)
    print("----")

    weights = np.ones(fun.nr)
    log_cost = proxddp.LogResidualCost(fun, weights)
    data = log_cost.createData()
    print(data)
    assert isinstance(data, proxddp.CompositeCostData)

    log_cost.evaluate(x0, u0, data)
    log_cost.computeGradients(x0, u0, data)
    log_cost.computeHessians(x0, u0, data)
    print("LogCost:")
    print(data.value)
    print(data.grad)
    print(data.hess)
    print("----")


def test_quad_state():
    space = manifolds.SE2()
    ndx = space.ndx
    nu = space.ndx
    x0 = space.neutral()
    x1 = space.rand()
    weights = np.eye(ndx)
    cost = proxddp.QuadraticStateCost(space, nu, x0, weights)

    data = cost.createData()

    print("x0", x0)
    print("x1", x1)
    u0 = np.zeros(nu)

    cost.evaluate(x0, u0, data)
    print(data.value)
    assert data.value == 0.0
    cost.computeGradients(x0, u0, data)
    print(data.grad)
    cost.computeHessians(x0, u0, data)
    print(data.Lxx)

    e = space.difference(x1, x0)
    print("err:", e)
    cost.evaluate(x1, u0, data)
    print(data.value)
    cost.computeGradients(x1, u0, data)
    print(data.Lx)
    cost.computeHessians(x1, u0, data)
    print(data.Lxx)


def test_stack_error():
    # Should raise RuntimeError due to wrong use.
    nx = 2
    nu = 2
    cost_stack = CostStack(nx, nu)
    Q = np.eye(nx)
    R = np.eye(nu)
    R[range(nu), range(nu)] = np.random.rand(nu) * 2.0
    rcost = QuadraticCost(Q, R)
    cost_stack.addCost(rcost)  # optional

    rc2 = QuadraticCost(np.eye(3), np.eye(nu))
    rc3 = QuadraticCost(np.eye(nx), np.eye(nu * 2))

    cost_data = cost_stack.createData()
    print(cost_data.sub_cost_data.tolist())

    with pytest.raises(Exception) as e_info:
        cost_stack.addCost(rc2)
    print(e_info)

    with pytest.raises(Exception) as e_info:
        cost_stack.addCost(rc3)
    print(e_info)

    with pytest.raises(Exception) as e_info:
        CostStack(nx, nu, [rcost, rc2], [1.0, 1.0])
    print(e_info)

    with pytest.raises(Exception) as e_info:
        CostStack(nx, nu, [rcost], [1.0, 1.0])
    print(e_info)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
