from proxnlp import manifolds
import proxddp
import numpy as np
from . import utils

import matplotlib.pyplot as plt

space = manifolds.SE2()
nu = space.ndx


class TwistModel(proxddp.DynamicsModel):
    def __init__(self, B: np.ndarray = None):
        if B is None:
            B = np.eye(nu)
        proxddp.DynamicsModel.__init__(self, space.ndx, nu)
        self.B = B

    def evaluate(self, x, u, y, data: proxddp.FunctionData):
        data.value[:] = space.difference(
            y, space.integrate(x, self.B @ u))

    def computeJacobians(self, x, u, y, data: proxddp.FunctionData):
        v_ = self.B @ u

        xnext = space.integrate(x, v_)

        Jv_u = self.B
        Jxnext_x = space.Jintegrate(x, v_, 0)
        Jxnext_v = space.Jintegrate(x, v_, 1)

        # res = space.difference(xnext, y)

        Jres_xnext = np.eye(space.ndx)
        Jres_y = np.eye(space.ndx)
        Jres_xnext = space.Jdifference(y, xnext, 0)
        Jres_y = space.Jdifference(y, xnext, 1)

        data.Jx[:, :] = Jres_xnext @ Jxnext_x
        data.Ju[:, :] = Jres_xnext @ Jxnext_v @ Jv_u
        data.Jy[:, :] = Jres_y


dynmodel = TwistModel()
data = dynmodel.createData()

x0 = space.rand()
u0 = np.random.randn(nu)
x1 = space.rand()

dynmodel.evaluate(x0, u0, x1, data)
dynmodel.computeJacobians(x0, u0, x1, data)


stage_model = proxddp.StageModel(space, nu, dynmodel)
stage_data = stage_model.createData()

shooting_problem = proxddp.ShootingProblem()
shooting_problem.add_stage(stage_model)


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


us_ = [u0] * 20
print("const control u0:", u0)
expdyn = TwistModelExplicit(dt=0.1)
xs_out = proxddp.rollout(expdyn, x0, us_).tolist()

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
