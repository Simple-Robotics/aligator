from aligator import manifolds, QuadraticCost, CostStack
import aligator
import numpy as np
import eigenpy
import pytest

from numpy.testing import assert_allclose
from utils import cost_finite_grad

FD_EPS = 1e-7
ATOL = 2 * FD_EPS**0.5


def sample_gauss(space):
    x0 = space.neutral()
    d = np.random.randn(space.ndx) * 0.1
    return space.integrate(x0, d)


def test_cost_stack():
    nx = 2
    nu = 2
    space = manifolds.VectorSpace(nx)
    cost_stack = CostStack(space, nu)
    Qr = np.random.randn(4, nx)
    Q = Qr.T @ Qr / nx
    R = np.eye(nu)
    rcost = QuadraticCost(Q, R)
    assert isinstance(rcost.space, manifolds.VectorSpace)

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
        assert_allclose(data1.grad, data2.grad, atol=ATOL)
        assert_allclose(data1.hess, data2.hess, atol=ATOL)

        assert_allclose(data1.Lxx, Q, atol=ATOL)
        assert_allclose(data1.Luu, R, atol=ATOL)

    with pytest.raises(KeyError, match="Key unk not found."):
        cost_stack.getComponent("unk")

    rcost_ref = cost_stack.getComponent(0)
    assert isinstance(rcost_ref, QuadraticCost)
    rcost_ref.interp_u[:] = 0.1

    rcost_ref2 = cost_stack.getComponent(0)
    assert isinstance(rcost_ref2, QuadraticCost)
    assert_allclose(rcost_ref2.interp_u, 0.1)

    # second getter API
    rcost_ref3, w = cost_stack.components[0]
    assert isinstance(rcost_ref3, QuadraticCost)
    assert w == 1.0

    cost_stack.setWeight(0, 2.0)
    assert cost_stack.components[0][1] == 2.0
    # check that rcost_ref3, using map getter API, was indeed a reference
    rcost_ref3.interp_x[:] = 0.42
    assert_allclose(rcost_ref.interp_x, 0.42)  # fails

    # test other API for cost,
    # building from dict
    cost_stack = CostStack(space, nu, {"quad": (rcost, 1.0)})
    assert cost_stack.size == 1


def test_composite_cost():
    space = manifolds.VectorSpace(3)
    ndx = space.ndx
    x0 = space.rand()

    nu = space.ndx
    u0 = np.ones(nu)
    target = x0 - 0.1 * np.ones(ndx)
    fun = aligator.StateErrorResidual(space, nu, target)
    # for debug
    fd = fun.createData()
    fun.evaluate(x0, u0, fd)
    fun.computeJacobians(x0, u0, fd)

    # costs
    weights = np.random.randn(4, fun.nr)
    weights = weights.T @ weights
    cost = aligator.QuadraticResidualCost(space, fun, weights)
    assert np.array_equal(weights, cost.weights)

    data = cost.createData()
    print("Composite data:", data)
    assert isinstance(data, aligator.CompositeCostData)

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
    assert_allclose(data.grad, ref_grad)
    assert_allclose(data.hess, ref_hess)

    weights = np.ones(fun.nr)
    log_cost = aligator.LogResidualCost(space, fun, weights)
    data: aligator.CompositeCostData = log_cost.createData()
    print(data)
    assert isinstance(data, aligator.CompositeCostData)

    log_cost.evaluate(x0, u0, data)
    log_cost.computeGradients(x0, u0, data)
    log_cost.computeHessians(x0, u0, data)
    print("LogCost:")
    print(data.value)
    print(data.residual_data.value)
    print(data.grad)
    print(data.hess)
    for i in range(100):
        x0 = sample_gauss(space)
        cost.evaluate(x0, u0, data)
        cost.computeGradients(x0, u0, data)
        fgrad = cost_finite_grad(cost, space, x0, u0, FD_EPS)
        assert_allclose(fgrad, data.grad, atol=ATOL)


def test_log_barrier():
    space = manifolds.VectorSpace(3)
    x0 = space.rand()

    nu = space.ndx
    u0 = np.ones(nu)
    fun = aligator.StateErrorResidual(space, nu, target=space.rand())
    # for debug
    fd = fun.createData()
    fun.evaluate(x0, u0, fd)
    fun.computeJacobians(x0, u0, fd)

    # costs
    weights = np.ones(fun.nr)
    thresh = np.random.rand()
    cost = aligator.RelaxedLogBarrierCost(space, fun, weights, thresh)
    assert np.array_equal(weights, cost.weights)

    data = cost.createData()
    print("Composite data:", data)
    assert isinstance(data, aligator.CompositeCostData)

    cost.evaluate(x0, u0, data)
    cost.computeGradients(x0, u0, data)
    cost.computeHessians(x0, u0, data)

    print("RelaxedLogCost:")
    print(data.value)
    print(data.grad)
    print(data.hess)

    for _ in range(100):
        x0 = sample_gauss(space)
        cost.evaluate(x0, u0, data)
        cost.computeGradients(x0, u0, data)
        fgrad = cost_finite_grad(cost, space, x0, u0, FD_EPS)
        assert_allclose(fgrad, data.grad, atol=ATOL)


def test_quad_state():
    space = manifolds.SE2()
    ndx = space.ndx
    nu = space.ndx
    x0 = space.neutral()
    x1 = space.rand()
    weights = np.eye(ndx)
    cost = aligator.QuadraticStateCost(space, nu, x0, weights)

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
    space = manifolds.VectorSpace(nx)
    cost_stack = CostStack(space, nu)
    Q = np.eye(nx)
    R = np.eye(nu)
    R[range(nu), range(nu)] = np.random.rand(nu) * 2.0
    rcost = QuadraticCost(Q, R)
    cost_stack.addCost(rcost)  # optional

    if eigenpy.__version__ >= "3.9.1":
        print(cost_stack.components.todict())

    rc2 = QuadraticCost(np.eye(3), np.eye(nu))
    rc3 = QuadraticCost(np.eye(nx), np.eye(nu * 2))

    if eigenpy.__version__ >= "3.9.1":
        cost_data = cost_stack.createData()
        print(cost_data.sub_cost_data.todict())

    with pytest.raises(Exception) as e_info:
        cost_stack.addCost(rc2)
    print(e_info)

    with pytest.raises(Exception) as e_info:
        cost_stack.addCost(rc3)
    print(e_info)

    with pytest.raises(Exception) as e_info:
        CostStack(space, nu, [rcost, rc2], [1.0, 1.0])
    print(e_info)

    with pytest.raises(Exception) as e_info:
        CostStack(space, nu, [rcost], [1.0, 1.0])
    print(e_info)


def test_direct_sum():
    import pinocchio as pin

    model: pin.Model = pin.buildSampleModelManipulator()
    data: pin.Data = model.createData()
    config_space = manifolds.MultibodyConfiguration(model)
    nv = config_space.ndx
    frame_name = "effector_body"
    frame_id = model.getFrameId(frame_name)
    q0 = config_space.neutral()
    pin.framesForwardKinematics(model, data, q0)
    p_ref = data.oMf[frame_id].translation
    frame_fn = aligator.FrameTranslationResidual(
        config_space.ndx, nv, model, p_ref, frame_id
    )
    print(frame_fn.nr)
    frame_cost = aligator.QuadraticResidualCost(config_space, frame_fn, np.eye(3))

    cam_space = manifolds.SE3()
    cam_cost = aligator.QuadraticStateCost(
        cam_space, cam_space.ndx, cam_space.neutral(), np.eye(6) * 0.01
    )

    # direct sum
    direct_sum = aligator.directSum(cam_cost, frame_cost)
    data = direct_sum.createData()
    assert isinstance(data, aligator.DirectSumCostData)
    d1 = data.data1
    d2 = data.data2
    space = direct_sum.space
    assert isinstance(space, manifolds.CartesianProduct)
    assert space.num_components == 2

    x0 = space.rand()
    u0 = np.random.randn(direct_sum.nu)
    direct_sum.evaluate(x0, u0, data)
    direct_sum.computeGradients(x0, u0, data)
    np.set_printoptions(precision=5, linewidth=250)
    print(data.Lx)
    print(d1.Lx, d2.Lx)
    assert_allclose(data.value, d1.value + d2.value)
    assert_allclose(data.Lx[:nv], d1.Lx)
    assert_allclose(data.Lx[nv:], d2.Lx)


if __name__ == "__main__":
    import sys

    SEED = 40
    np.random.seed(SEED)
    aligator.seed(SEED)

    sys.exit(pytest.main(sys.argv))
