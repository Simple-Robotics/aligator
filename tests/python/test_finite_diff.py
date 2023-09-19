from proxddp import (
    FiniteDifferenceHelper,
    CostFiniteDifference,
    QuadraticStateCost,
    ControlBoxFunction,
    StateErrorResidual,
    manifolds,
)
import numpy as np


def test_compute_jac_vs():
    nx = 2
    nu = 5
    space = manifolds.VectorSpace(nx)
    u_min, u_max = -np.ones(nu), np.ones(nu)
    fun1 = ControlBoxFunction(nx, u_min, u_max)
    fdata1 = fun1.createData()
    fun1_fd = FiniteDifferenceHelper(space, fun1, 1e-3)
    fdata1b = fun1_fd.createData()
    for i in range(100):
        x0 = np.random.random(nx)
        u0 = 0.6 * np.ones(nu)
        fun1.evaluate(x0, u0, x0, fdata1)
        fun1_fd.evaluate(x0, u0, x0, fdata1b)
        assert np.allclose(fdata1.value, fdata1b.value, 1e-2)
        fun1.computeJacobians(x0, u0, x0, fdata1)
        fun1_fd.computeJacobians(x0, u0, x0, fdata1b)
        assert np.allclose(fdata1.Jx, fdata1b.Jx, 1e-2)
        assert np.allclose(fdata1.Ju, fdata1b.Ju, 1e-2)
        assert np.allclose(fdata1.Jy, fdata1b.Jy, 1e-2)


def test_compute_jac_multibody():
    try:
        import pinocchio as pin

        model = pin.buildSampleModelHumanoid()
        space = manifolds.MultibodyConfiguration(model)
        nu = 3
        x_tar = space.neutral()
        fun2 = StateErrorResidual(space, nu, x_tar)
        fdata2 = fun2.createData()
        fun2_fd = FiniteDifferenceHelper(space, fun2, 1e-6)
        fdata2b = fun2_fd.createData()
        for i in range(1000):
            x0 = pin.randomConfiguration(model, -np.ones(model.nq), np.ones(model.nq))
            u0 = 0.6 * np.ones(nu)
            fun2.evaluate(x0, u0, x0, fdata2)
            fun2_fd.evaluate(x0, u0, x0, fdata2b)
            assert np.allclose(fdata2.value, fdata2b.value, 1e-2)
            fun2.computeJacobians(x0, u0, x0, fdata2)
            fun2_fd.computeJacobians(x0, u0, x0, fdata2b)
            assert np.allclose(fdata2.Jx, fdata2b.Jx, 1e-2)
            assert np.allclose(fdata2.Ju, fdata2b.Ju, 1e-2)
            assert np.allclose(fdata2.Jy, fdata2b.Jy, 1e-2)
        return
    except ImportError:
        pass


def test_compute_cost_se3():
    space = manifolds.SE3()
    x0 = space.neutral()
    weights = np.eye(space.ndx)
    cost = QuadraticStateCost(space, space.ndx, x0, weights)
    data = cost.createData()
    cost_fd = CostFiniteDifference(cost, fd_eps=1e-6)
    data_fd = cost_fd.createData()
    for i in range(100):
        x0 = space.rand()
        u0 = 0.6 * np.ones(space.ndx)
        cost.evaluate(x0, u0, data)
        cost_fd.evaluate(x0, u0, data_fd)
        assert np.allclose(data.value, data_fd.value, 1e-2)
        cost.computeGradients(x0, u0, data)
        cost_fd.computeGradients(x0, u0, data_fd)
        assert np.allclose(data.Lx, data_fd.Lx, 1e-2)
        assert np.allclose(data.Lu, data_fd.Lu, 1e-2)



if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(sys.argv))
