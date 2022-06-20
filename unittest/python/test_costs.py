from proxddp import QuadraticCost, CostStack
import numpy as np


def test_cost_stack():
    nx = 2
    nu = 2
    cost_stack = CostStack(nx, nu)
    Q = np.eye(nx)
    R = np.eye(nu)
    rcost = QuadraticCost(Q, R)
    print(rcost)

    cost_stack.addCost(rcost, 1.0)
    data1 = rcost.createData()
    data2 = cost_stack.createData()

    x0 = np.random.randn(nx)
    u0 = np.random.randn(nu)

    rcost.evaluate(x0, u0, data1)
    cost_stack.evaluate(x0, u0, data2)

    print(data1.value)
    print(data2.value)

    rcost.computeGradients(x0, u0, data1)
    cost_stack.computeGradients(x0, u0, data2)

    rcost.computeHessians(x0, u0, data1)
    cost_stack.computeHessians(x0, u0, data2)

    assert data1.value == data2.value
    assert np.allclose(data1._grad, data2._grad)
    assert np.allclose(data1._hessian, data2._hessian)
