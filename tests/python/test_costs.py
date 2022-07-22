from proxddp import QuadraticCost, CostStack
from proxddp import manifolds
import proxddp
import numpy as np

import pytest


def test_cost_stack():
    nx = 2
    nu = 2
    cost_stack = CostStack(nx, nu)
    Q = np.eye(nx)
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


def test_composite_cost():
    space = manifolds.SE2()
    nu = space.ndx
    target = space.rand()
    fun = proxddp.StateErrorResidual(space, nu, target)
    weights = np.eye(fun.nr)
    cost = proxddp.QuadraticResidualCost(fun, weights) 

    data = cost.createData()
    print("Composite data:", data)
    assert isinstance(data, proxddp.CompositeCostData)


# Should raise RuntimeError due to wrong use.
def test_stack_error():
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
