from proxnlp import manifolds
import proxddp
import numpy as np
from . import utils

import matplotlib.pyplot as plt

space = manifolds.SE2()
nu = space.ndx


x0 = space.rand()
u0 = np.random.randn(nu)
x1 = space.rand()


class TwistModelExplicit(proxddp.ExplicitDynamicsModel):
    def __init__(self, dt: float, B: np.ndarray = None):
        if B is None:
            B = np.eye(nu)
        self.B = B
        self.dt = dt
        super().__init__(space, nu)

    def forward(self, x, u, out):
        out[:] = space.integrate(x, self.dt * self.B @ u)

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

    def evaluate(self, x, u, data):
        dx = space.difference(x, self.x_ref)
        data.value = 0.5 * np.dot(dx, self.W @ dx)

    def computeGradients(self, x, u, data):
        space.Jdifference(x, self.x_ref, data.Lx, 0)
        data.Lx[:] = self.W @ data.Lx
        data.Lu[:] = 0.

    def computeHessians(self, x, u, data):
        J = space.Jdifference(x, self.x_ref, 0)
        data._hessian[:] = 0.
        data.Lxx[:, :] = J.T @ self.W @ J


us_ = [u0] * 20
print("const control u0:", u0)
dynmodel = TwistModelExplicit(dt=0.1)
dyn_data = dynmodel.createData()
xs_out = proxddp.rollout(dynmodel, x0, us_).tolist()

dynmodel.evaluate(x0, u0, x1, dyn_data)
dynmodel.computeJacobians(x0, u0, x1, dyn_data)

cost = MyQuadCost(W=np.eye(space.ndx), x_ref=x1)

stage_model = proxddp.StageModel(space, nu, cost, dynmodel)
stage_data = stage_model.createData()
cost_data = stage_data.cost_data

cost.evaluate(x0, u0, cost_data)
cost.computeGradients(x0, u0, cost_data)
cost.computeHessians(x0, u0, cost_data)

shooting_problem = proxddp.ShootingProblem()
shooting_problem.add_stage(stage_model)


fig, ax = plt.subplots()
ax: plt.Axes
cmap = plt.get_cmap("viridis")
cols_ = cmap(np.linspace(0, 1, len(xs_out)))

for i, q in enumerate(xs_out):
    utils.plot_se2_pose(q, ax, alpha=0.2, fc=cols_[i])
ax.autoscale_view()
ax.set_title("Motion in $\\mathrm{SE}(2)$")

ax.set_aspect("equal")
plt.show()
