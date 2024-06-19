"""

You can run this file using pytest:

    pytest examples/se2_twist.py -s

Or also as a module.
"""

from proxsuite_nlp import costs
import aligator
from aligator import manifolds, CommonModelDataContainer
import numpy as np
import typing

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
        self.is_configured = False
        super().__init__(space, nu)

    def configure(self, container):
        self.is_configured = True

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

    def createData(self, container):
        return TwistData()


class TwistData(aligator.dynamics.ExplicitDynamicsData):
    def __init__(self):
        super().__init__(ndx, nu, nx, ndx)
        self.good = True


class MyCostData(aligator.CostData):
    def __init__(self):
        super().__init__(space.ndx, nu)


class MyQuadCost(aligator.CostAbstract):
    def __init__(self, W: np.ndarray, x_ref: np.ndarray):
        self.x_ref = x_ref
        self.W = W
        super().__init__(space, nu)
        self._basis = costs.QuadraticDistanceCost(space, self.x_ref, self.W)
        self.is_configured = False

    def configure(self, container):
        self.is_configured = True

    def evaluate(self, x, u, data):
        assert isinstance(data, MyCostData)
        data.value = self._basis.call(x)

    def computeGradients(self, x, u, data):
        assert isinstance(data, MyCostData)
        self._basis.computeGradient(x, data.Lx)
        data.Lu[:] = 0.0

    def computeHessians(self, x, u, data):
        assert isinstance(data, MyCostData)
        self._basis.computeGradient(x, data.Lx)
        data.hess[:, :] = 0.0
        self._basis.computeHessian(x, data.Lxx)

    def createData(self, container):
        return MyCostData()


DT = 0.1


class Problem(typing.NamedTuple):
    dynmodel: TwistModelExplicit
    cost: MyQuadCost
    stage_model: aligator.StageModel


@pytest.fixture
def problem():
    dynmodel = TwistModelExplicit(DT)
    cost = MyQuadCost(W=np.eye(space.ndx), x_ref=x1)
    stage_model = aligator.StageModel(cost, dynmodel)
    return Problem(dynmodel, cost, stage_model)


@pytest.mark.parametrize("nsteps", [1, 4])
class TestClass:
    dt = DT
    x0 = space.neutral()
    tol = 1e-5
    mu_init = 1e-2
    solver = aligator.SolverProxDDP(tol, mu_init, 0.0)

    def test_dyn(self, nsteps, problem):
        dyn_data = problem.dynmodel.createData(CommonModelDataContainer())
        assert isinstance(dyn_data, TwistData)
        dyn_data.Jx[:, :] = np.arange(ndx**2).reshape(ndx, ndx)
        dyn_data.Ju[:, :] = np.arange(ndx**2, ndx**2 + ndx * nu).reshape(ndx, nu)
        problem.dynmodel.evaluate(x0, u0, x1, dyn_data)
        problem.dynmodel.computeJacobians(x0, u0, x1, dyn_data)

    def test_cost(self, nsteps, problem):
        cost = problem.cost
        cost_data = cost.createData(CommonModelDataContainer())
        cost.evaluate(x0, u0, cost_data)
        cost.computeGradients(x0, u0, cost_data)
        cost.computeHessians(x0, u0, cost_data)

    def test_stage(self, nsteps, problem):
        stage_model = problem.stage_model
        sd = stage_model.createData()
        stage_model.computeFirstOrderDerivatives(x0, u0, x1, sd)
        stage_model.num_primal == ndx + nu
        stage_model.num_dual == ndx

    def test_rollout(self, nsteps, problem):
        us_i = [np.ones(problem.dynmodel.nu) * 0.1 for _ in range(nsteps)]
        xs_i = aligator.rollout(problem.dynmodel, self.x0, us_i).tolist()
        dd = problem.dynmodel.createData(CommonModelDataContainer())
        problem.dynmodel.forward(self.x0, us_i[0], dd)
        assert np.allclose(dd.xnext, xs_i[1])

    def test_shooting_problem(self, nsteps, problem):
        stage_model = problem.stage_model
        opt_problem = aligator.TrajOptProblem(
            self.x0, nu, space, term_cost=problem.cost
        )
        for _ in range(nsteps):
            opt_problem.addStage(stage_model)

        opt_problem_data = aligator.TrajOptData(opt_problem)
        stage_datas = opt_problem_data.stage_data

        print("term cost data:", opt_problem_data.term_cost)
        print("term cstr data:", opt_problem_data.term_constraint)

        stage2 = stage_model.clone()
        sd0 = stage_datas[0].clone()
        print("Clone stage:", stage2)
        print("Clone stage data:", sd0)

        us_init = [u0] * nsteps
        xs_out = aligator.rollout(problem.dynmodel, x0, us_init).tolist()

        assert len(opt_problem_data.stage_data) == opt_problem.num_steps
        assert opt_problem.num_steps == nsteps

        opt_problem.evaluate(xs_out, us_init, opt_problem_data)
        opt_problem.computeDerivatives(xs_out, us_init, opt_problem_data)

        solver = self.solver

        assert solver.bcl_params.prim_alpha == 0.1
        assert solver.bcl_params.prim_beta == 0.9
        assert solver.bcl_params.dual_alpha == 1.0
        assert solver.bcl_params.dual_beta == 1.0

        solver.multiplier_update_mode = aligator.MultiplierUpdateMode.NEWTON
        solver.setup(problem)
        assert problem.dynmodel.is_configured
        assert problem.cost.is_configured
        solver.run(problem, xs_out, us_init)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
