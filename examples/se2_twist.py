from proxnlp import manifolds
import proxddp
import numpy as np


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
        print("In:")
        print(Jx, "x")
        print(Ju, "u")
        v_ = self.dt * self.B @ u
        dv_du = self.dt * self.B

        space.Jintegrate(x, v_, Jx, 0)
        Jxnext_dv = space.Jintegrate(x, v_, 1)
        Ju[:, :] = Jxnext_dv @ dv_du
        print("Out:")
        print(Jx, "x")
        print(Ju, "u")
        print("done dForward()")


class MyQuadCost(proxddp.CostBase):
    def __init__(self, W: np.ndarray, x_ref: np.ndarray):
        self.x_ref = x_ref
        self.W = W
        super().__init__(space.ndx, nu)

    def evaluate(self, x, u, data):
        dx = np.zeros(ndx)
        space.difference(x, self.x_ref, dx)
        data.value = 0.5 * np.dot(dx, self.W @ dx)

    def computeGradients(self, x, u, data):
        dx = space.difference(x, self.x_ref)
        J = space.Jdifference(x, self.x_ref, 0)
        data.Lx[:] = J.T @ self.W @ dx
        data.Lu[:] = 0.

    def computeHessians(self, x, u, data):
        J = space.Jdifference(x, self.x_ref, 0)
        data._hessian[:, :] = 0.
        data.Lxx[:, :] = J.T @ self.W @ J


nsteps = 1
us_ = [u0] * nsteps
print("const control u0:", u0)
dynmodel = TwistModelExplicit(dt=0.1)
dyn_data = dynmodel.createData()
xs_out = proxddp.rollout(dynmodel, x0, us_).tolist()
print("xs_out:")
print(xs_out)
cost = MyQuadCost(W=np.eye(space.ndx), x_ref=x1)


def test_dyn_cost():

    dyn_data.Jx[:, :] = np.arange(ndx ** 2).reshape(ndx, ndx)
    dyn_data.Ju[:, :] = np.arange(ndx ** 2, ndx ** 2 + ndx * nu).reshape(ndx, nu)
    dynmodel.evaluate(x0, u0, x1, dyn_data)
    dynmodel.computeJacobians(x0, u0, x1, dyn_data)
    print("dynmodel after:")
    print(dyn_data.Jx, "x")
    print(dyn_data.Ju, "u")

    cost_data = cost.createData()
    cost.evaluate(x0, u0, cost_data)
    cost.computeGradients(x0, u0, cost_data)
    cost.computeHessians(x0, u0, cost_data)


test_dyn_cost()

stage_model = proxddp.StageModel(space, nu, cost, dynmodel)
sd = stage_model.createData()
sd.dyn_data.Jx[:, :] = np.arange(ndx * ndx).reshape(ndx, ndx)
stage_model.computeDerivatives(x0, u0, x1, sd)
print(sd.dyn_data.Jx, "after")


# Define shooting problem

def test_shooting_problem():
    stage_model = proxddp.StageModel(space, nu, cost, dynmodel)
    shooting_problem = proxddp.ShootingProblem()
    for _ in range(nsteps):
        shooting_problem.add_stage(stage_model)

    problem_data = shooting_problem.createData()
    stage_datas = problem_data.stage_data
    stage_datas[0].dyn_data.Jx[:, :] = np.arange(ndx * ndx).reshape(ndx, ndx)
    # print(stage_datas[0].dyn_data.Jx, "dd0 Jx")
    print(stage_datas[0].dyn_data.Jx, "dd0 Jx")

    assert len(problem_data.stage_data) == shooting_problem.num_steps
    assert shooting_problem.num_steps == nsteps

    shooting_problem.evaluate(xs_out, us_, problem_data)
    shooting_problem.computeDerivatives(xs_out, us_, problem_data)


test_shooting_problem()

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
