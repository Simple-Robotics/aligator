"""

You can run this file using pytest:

    pytest examples/se2_twist.py -s

Or also as a module.
"""

import aligator
from aligator import manifolds
import numpy as np

import pytest


np.set_printoptions(precision=4, linewidth=250)

space = manifolds.SE2()
nx = space.nx
ndx = space.ndx
nu = space.ndx


np.random.seed(42)  # of course
x0 = space.rand()
u0 = np.random.randn(nu)
x1 = space.neutral()


class TwistModelExplicit(aligator.dynamics.ExplicitDynamicsModel):
    def __init__(self, dt: float, B: np.ndarray = None):
        if B is None:
            B = np.eye(nu)
        self.B = B
        self.dt = dt
        super().__init__(space, nu)

    def __getinitargs__(self):
        return (self.dt, self.B)

    def forward(self, x, u, data: aligator.dynamics.ExplicitDynamicsData):
        assert data.good
        space.integrate(x, self.dt * self.B @ u, data.xnext)

    def dForward(self, x, u, data: aligator.dynamics.ExplicitDynamicsData):
        Jx = data.Jx
        Ju = data.Ju
        v_ = self.dt * self.B @ u
        dv_du = self.dt * self.B

        space.Jintegrate(x, v_, Jx, 0)
        Jxnext_dv = space.Jintegrate(x, v_, 1)
        Ju[:, :] = Jxnext_dv @ dv_du

    def createData(self):
        return TwistData(self)


class TwistData(aligator.dynamics.ExplicitDynamicsData):
    def __init__(self, model):
        super().__init__(model)
        self.good = True


class MyCostData(aligator.CostData):
    def __init__(self):
        super().__init__(space.ndx, nu)


def _call(space: manifolds.ManifoldAbstract, x_ref, x, W):
    err = space.difference(x_ref, x)
    return err.dot(W @ err)


def _jac(space: manifolds.ManifoldAbstract, x_ref, x, W):
    err = space.difference(x_ref, x)
    J = space.Jdifference(x_ref, x, 1)
    return J.T @ (W @ err)


def _hess(space: manifolds.ManifoldAbstract, x_ref, x, W):
    J = space.Jdifference(x_ref, x, 1)
    return J.T @ (W @ J)


class MyQuadCost(aligator.CostAbstract):
    def __init__(self, W: np.ndarray, x_ref: np.ndarray):
        self.x_ref = x_ref
        self.W = W
        super().__init__(space, nu)

    def __getinitargs__(self):
        return (self.W, self.x_ref)

    def __getstate__(self):
        return dict()

    def evaluate(self, x, u, data):
        assert isinstance(data, MyCostData)
        data.value = _call(space, self.x_ref, x, self.W)

    def computeGradients(self, x, u, data):
        assert isinstance(data, MyCostData)
        data.Lx[:] = _jac(space, self.x_ref, x, self.W)
        data.Lu[:] = 0.0

    def computeHessians(self, x, u, data):
        assert isinstance(data, MyCostData)
        data.hess[:, :] = 0.0
        data.Lxx[:, :] = _hess(space, self.x_ref, x, self.W)

    def createData(self):
        return MyCostData()


@pytest.mark.parametrize("nsteps", [1, 4])
class TestClass:
    dt = 0.1
    x0 = space.neutral()
    dyn_model = TwistModelExplicit(dt)
    cost = MyQuadCost(W=np.eye(space.ndx), x_ref=x1)
    stage_model = aligator.StageModel(cost, dyn_model)
    tol = 1e-5
    mu_init = 1e-2
    solver: aligator.SolverProxDDP = aligator.SolverProxDDP(tol, mu_init)

    def test_dyn(self, nsteps):
        dyn_data = self.dyn_model.createData()
        assert isinstance(dyn_data, TwistData)
        dyn_data.Jx[:, :] = np.arange(ndx**2).reshape(ndx, ndx)
        dyn_data.Ju[:, :] = np.arange(ndx**2, ndx**2 + ndx * nu).reshape(ndx, nu)
        self.dyn_model.forward(x0, u0, dyn_data)
        self.dyn_model.dForward(x0, u0, dyn_data)
        print(self.stage_model.dynamics)
        assert isinstance(self.stage_model.dynamics, TwistModelExplicit)

    def test_cost(self, nsteps):
        cost = self.cost
        cost_data = cost.createData()
        cost.evaluate(x0, u0, cost_data)
        cost.computeGradients(x0, u0, cost_data)
        cost.computeHessians(x0, u0, cost_data)
        assert isinstance(self.stage_model.cost, MyQuadCost)

    def test_stage(self, nsteps):
        stage_model = self.stage_model
        sd = stage_model.createData()
        stage_model.computeFirstOrderDerivatives(x0, u0, sd)
        stage_model.num_dual == ndx

    def test_rollout(self, nsteps):
        us_i = [np.ones(self.dyn_model.nu) * 0.1 for _ in range(nsteps)]
        xs_i = aligator.rollout(self.dyn_model, self.x0, us_i).tolist()
        dd = self.dyn_model.createData()
        self.dyn_model.forward(self.x0, us_i[0], dd)
        assert np.allclose(dd.xnext, xs_i[1])

    def test_shooting_problem(self, nsteps):
        stage_model = self.stage_model
        problem = aligator.TrajOptProblem(self.x0, nu, space, term_cost=self.cost)
        for _ in range(nsteps):
            problem.addStage(stage_model)

        problem_data = aligator.TrajOptData(problem)

        print("term cost data:", problem_data.term_cost)
        print("term cstr data:", problem_data.term_constraint)

        stage2 = stage_model.copy()
        sd0 = stage2.createData()
        print("Clone stage:", stage2)
        print("Clone stage data:", sd0)

        us_init = [u0] * nsteps
        xs_out = aligator.rollout(self.dyn_model, x0, us_init).tolist()

        assert len(problem_data.stage_data) == problem.num_steps
        assert problem.num_steps == nsteps

        problem.evaluate(xs_out, us_init, problem_data)
        problem.computeDerivatives(xs_out, us_init, problem_data)

        solver = self.solver

        assert solver.bcl_params.prim_alpha == 0.1
        assert solver.bcl_params.prim_beta == 0.9
        assert solver.bcl_params.dual_alpha == 1.0
        assert solver.bcl_params.dual_beta == 1.0

        solver.setup(problem)
        solver.rollout_type = aligator.ROLLOUT_LINEAR
        solver.run(problem, xs_out, us_init)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
