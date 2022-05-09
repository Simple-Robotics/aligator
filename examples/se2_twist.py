from proxnlp import manifolds
from proxnlp import costs
import proxddp
import numpy as np

import pytest


np.set_printoptions(precision=4, linewidth=250)

space = manifolds.SE2()
ndx = space.ndx
nu = space.ndx


x0 = space.rand()
u0 = np.random.randn(nu)
x1 = space.neutral()


class TwistModelExplicit(proxddp.ExplicitDynamicsModel):
    def __init__(self, dt: float, B: np.ndarray = None):
        if B is None:
            B = np.eye(nu)
        self.B = B
        self.dt = dt
        super().__init__(space, nu)

    def forward(self, x, u, out):
        space.integrate(x, self.dt * self.B @ u, out)

    def dForward(self, x, u, Jx, Ju):
        v_ = self.dt * self.B @ u
        dv_du = self.dt * self.B

        space.Jintegrate(x, v_, Jx, 0)
        Jxnext_dv = space.Jintegrate(x, v_, 1)
        Ju[:, :] = Jxnext_dv @ dv_du


class MyQuadCost(proxddp.CostBase):
    def __init__(self, W: np.ndarray, x_ref: np.ndarray):
        self.x_ref = x_ref
        self.W = W
        super().__init__(space.ndx, nu)
        self._basis = costs.QuadraticDistanceCost(space, self.x_ref, self.W)

    def evaluate(self, x, u, data):
        data.value = self._basis.call(x)

    def computeGradients(self, x, u, data):
        self._basis.computeGradient(x, data.Lx)
        data.Lu[:] = 0.

    def computeHessians(self, x, u, data):
        data._hessian[:, :] = 0.
        self._basis.computeHessian(x, data.Lxx)


@pytest.mark.parametrize("nsteps", [1, 4])
class TestClass:
    dt = 0.1
    dynmodel = TwistModelExplicit(dt)
    cost = MyQuadCost(W=np.eye(space.ndx), x_ref=x1)
    stage_model = proxddp.StageModel(space, nu, cost, dynmodel)

    def test_dyn(self, nsteps):
        dyn_data = self.dynmodel.createData()
        dyn_data.Jx[:, :] = np.arange(ndx ** 2).reshape(ndx, ndx)
        dyn_data.Ju[:, :] = np.arange(ndx ** 2, ndx ** 2 + ndx * nu).reshape(ndx, nu)
        self.dynmodel.evaluate(x0, u0, x1, dyn_data)
        self.dynmodel.computeJacobians(x0, u0, x1, dyn_data)
        print(dyn_data.Jx, "x")
        print(dyn_data.Ju, "u")

    def test_cost(self, nsteps):
        cost = self.cost
        cost_data = cost.createData()
        cost.evaluate(x0, u0, cost_data)
        cost.computeGradients(x0, u0, cost_data)
        cost.computeHessians(x0, u0, cost_data)

    def test_stage(self, nsteps):
        stage_model = self.stage_model
        sd = stage_model.createData()
        sd.dyn_data.Jx[:, :] = np.arange(ndx * ndx).reshape(ndx, ndx)
        stage_model.computeDerivatives(x0, u0, x1, sd)
        stage_model.num_primal == ndx + nu
        stage_model.num_dual == ndx
        print(sd.dyn_data.Jx, "after")

    def test_shooting_problem(self, nsteps):
        stage_model = self.stage_model
        shooting_problem = proxddp.ShootingProblem()
        for _ in range(nsteps):
            shooting_problem.add_stage(stage_model)

        problem_data = shooting_problem.createData()
        stage_datas = problem_data.stage_data
        stage_datas[0].dyn_data.Jx[:, :] = np.arange(ndx * ndx).reshape(ndx, ndx)
        print(stage_datas[0].dyn_data.Jx, "dd0 Jx")

        us_ = [u0] * nsteps
        xs_out = proxddp.rollout(self.dynmodel, x0, us_).tolist()

        assert len(problem_data.stage_data) == shooting_problem.num_steps
        assert shooting_problem.num_steps == nsteps

        shooting_problem.evaluate(xs_out, us_, problem_data)
        shooting_problem.computeDerivatives(xs_out, us_, problem_data)

        ws = proxddp.Workspace(shooting_problem)
        ws.gains
        assert ws.kkt_matrix_buffer_.shape[0] == stage_model.num_primal + stage_model.num_dual

# import matplotlib.pyplot as plt
# import utils

# fig, ax = plt.subplots()
# ax: plt.Axes
# cmap = plt.get_cmap("viridis")
# cols_ = cmap(np.linspace(0, 1, len(xs_out)))

# for i, q in enumerate(xs_out):
#     utils.plot_se2_pose(q, ax, alpha=0.2, fc=cols_[i])
# ax.autoscale_view()
# ax.set_title("Motion in $\\mathrm{SE}(2)$")

# ax.set_aspect("equal")
# plt.show()
