from proxddp import QuadraticCost, CostStack
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
        assert np.allclose(data1._grad, data2._grad)
        assert np.allclose(data1._hessian, data2._hessian)


# Should raise RuntimeError due to wrong use.
def test_stack_error():
    nx = 2
    nu = 2
    cost_stack = CostStack(nx, nu)
    Q = np.eye(nx)
    R = np.eye(nu)
    rcost = QuadraticCost(np.eye(nx), np.eye(nu))
    cost_stack.addCost(rcost)  # optional

    rc2 = QuadraticCost(np.eye(3), np.eye(nu))
    rc3 = QuadraticCost(np.eye(nx), np.eye(nu * 2))

    with pytest.raises(Exception) as e_info:
        cost_stack.addCost(rc2)
    print(e_info)

    with pytest.raises(Exception) as e_info:
        cost_stack.addCost(rc3)
    print(e_info)

    with pytest.raises(Exception) as e_info:
        CostStack(nx, nu, [rcost, rc2], [1., 1.])
    print(e_info)

    with pytest.raises(Exception) as e_info:
        CostStack(nx, nu, [rcost], [1., 1.])
    print(e_info)
