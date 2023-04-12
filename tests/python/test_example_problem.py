"""

You can run this file using pytest:

    pytest examples/se2_twist.py -s

Or also as a module.
"""
from proxnlp import costs
import proxddp
from proxddp import manifolds
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


class TwistModelExplicit(proxddp.dynamics.ExplicitDynamicsModel):
    def __init__(self, dt: float, B: np.ndarray = None):
        if B is None:
            B = np.eye(nu)
        self.B = B
        self.dt = dt
        super().__init__(space, nu)

    def forward(self, x, u, data: proxddp.dynamics.ExplicitDynamicsData):
        assert data.good
        space.integrate(x, self.dt * self.B @ u, data.xnext)

    def dForward(self, x, u, data: proxddp.dynamics.ExplicitDynamicsData):
        Jx = data.Jx
        Ju = data.Ju
        v_ = self.dt * self.B @ u
        dv_du = self.dt * self.B

        space.Jintegrate(x, v_, Jx, 0)
        Jxnext_dv = space.Jintegrate(x, v_, 1)
        Ju[:, :] = Jxnext_dv @ dv_du

    def createData(self):
        return TwistData()


class TwistData(proxddp.dynamics.ExplicitDynamicsData):
    def __init__(self):
        super().__init__(ndx, nu, nx, ndx)
        self.good = True


class MyQuadCost(proxddp.CostAbstract):
    def __init__(self, W: np.ndarray, x_ref: np.ndarray):
        self.x_ref = x_ref
        self.W = W
        super().__init__(space.ndx, nu)
        self._basis = costs.QuadraticDistanceCost(space, self.x_ref, self.W)

    def evaluate(self, x, u, data):
        data.value = self._basis.call(x)

    def computeGradients(self, x, u, data):
        self._basis.computeGradient(x, data.Lx)
        data.Lu[:] = 0.0

    def computeHessians(self, x, u, data):
        data.hess[:, :] = 0.0
        self._basis.computeHessian(x, data.Lxx)


@pytest.mark.parametrize("nsteps", [1, 4])
class TestClass:
    dt = 0.1
    x0 = space.neutral()
    dynmodel = TwistModelExplicit(dt)
    cost = MyQuadCost(W=np.eye(space.ndx), x_ref=x1)
    stage_model = proxddp.StageModel(cost, dynmodel)
    tol = 1e-5
    mu_init = 1e-2
    solver = proxddp.SolverProxDDP(tol, mu_init, 0.0)

    def test_ext_create_data(self, nsteps):
        import create_data_ext

        dyn_data = create_data_ext.my_create_data(self.dynmodel)
        assert isinstance(dyn_data, TwistData)
        print(dyn_data)

    def test_dyn(self, nsteps):
        dyn_data = self.dynmodel.createData()
        assert isinstance(dyn_data, TwistData)
        dyn_data.Jx[:, :] = np.arange(ndx**2).reshape(ndx, ndx)
        dyn_data.Ju[:, :] = np.arange(ndx**2, ndx**2 + ndx * nu).reshape(ndx, nu)
        self.dynmodel.evaluate(x0, u0, x1, dyn_data)
        self.dynmodel.computeJacobians(x0, u0, x1, dyn_data)

    def test_cost(self, nsteps):
        cost = self.cost
        cost_data = cost.createData()
        cost.evaluate(x0, u0, cost_data)
        cost.computeGradients(x0, u0, cost_data)
        cost.computeHessians(x0, u0, cost_data)

    def test_stage(self, nsteps):
        stage_model = self.stage_model
        sd = stage_model.createData()
        stage_model.computeDerivatives(x0, u0, x1, sd)
        stage_model.num_primal == ndx + nu
        stage_model.num_dual == ndx

    def test_rollout(self, nsteps):
        us_i = [np.ones(self.dynmodel.nu) * 0.1 for _ in range(nsteps)]
        xs_i = proxddp.rollout(self.dynmodel, self.x0, us_i).tolist()
        dd = self.dynmodel.createData()
        self.dynmodel.forward(self.x0, us_i[0], dd)
        assert np.allclose(dd.xnext, xs_i[1])

    def test_shooting_problem(self, nsteps):
        stage_model = self.stage_model
        problem = proxddp.TrajOptProblem(self.x0, nu, space, term_cost=self.cost)
        for _ in range(nsteps):
            problem.addStage(stage_model)

        problem_data = proxddp.TrajOptData(problem)
        stage_datas = problem_data.stage_data

        print("term cost data:", problem_data.term_cost)
        print("term cstr data:", problem_data.term_constraint)

        stage2 = stage_model.clone()
        sd0 = stage_datas[0].clone()
        print("Clone stage:", stage2)
        print("Clone stage data:", sd0)

        us_init = [u0] * nsteps
        xs_out = proxddp.rollout(self.dynmodel, x0, us_init).tolist()

        assert len(problem_data.stage_data) == problem.num_steps
        assert problem.num_steps == nsteps

        problem.evaluate(xs_out, us_init, problem_data)
        problem.computeDerivatives(xs_out, us_init, problem_data)

        solver = self.solver

        assert solver.bcl_params.prim_alpha == 0.1
        assert solver.bcl_params.prim_beta == 0.9
        assert solver.bcl_params.dual_alpha == 1.0
        assert solver.bcl_params.dual_beta == 1.0

        solver.multiplier_update_mode = proxddp.MultiplierUpdateMode.NEWTON
        solver.setup(problem)
        solver.setLinesearchMuLowerBound(1e-9)
        solver.run(problem, xs_out, us_init)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
