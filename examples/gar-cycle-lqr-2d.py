from proxddp import gar

import numpy as np

import matplotlib.pyplot as plt
from utils import ASSET_DIR


np.random.seed(42)
np.set_printoptions(precision=3, linewidth=250)

nx = 2
nu = 2


def random_unit(size):
    return np.random.uniform(-1.0, 1.0, size)


def create_knot(nx, nu):
    knot = gar.LQRKnot(nx, nu, 0)
    knot.Q[:] = np.eye(nx) * 0.01
    knot.R[:] = np.eye(nu) * 0.01
    knot.r[:] = random_unit(nu) * 0.01
    knot.A[:] = [[1.0, 0.1], [-0.1, 1.0]]
    knot.B[:] = np.eye(nx, nu)
    knot.E[:] = -np.eye(nx)
    knot.f[:] = random_unit(nx) * 0.1
    return knot


T = 10
base_knot = create_knot(nx, nu)
knots = [base_knot for _ in range(T)]
knots.append(create_knot(nx, 0))
prob = gar.LQRProblem(knots, 0)

PARAM_DIM = nx
prob.addParameterization(PARAM_DIM)

xf = np.array([0.05, 0.05])
kf = prob.stages[T]
kf.Q[:] = np.eye(nx) * 1.0
kf.q[:] = -kf.Q @ xf


def add_mid(t0, v):
    kt0 = prob.stages[t0]
    kt0.Q[:] = 0.05
    kt0.q[:] = -kt0.Q @ v


t0 = T // 2
xmid0 = np.array([0.1, 0.0])
add_mid(t0, xmid0)

print(prob.stages[0])
print(prob.stages[T])
prob.stages[0].Gx[:] = +np.eye(nx)
prob.stages[T].Gx[:] = -np.eye(nx)

solver = gar.ProximalRiccatiSolver(prob)
mu = 1e-8
solver.backward(mu, mu)

soltraj_ = gar.lqrInitializeSolution(prob)

th_grad = solver.thGrad
th_hess = solver.thHess
print("thGrad: {}".format(th_grad))
print("thHess:\n{}".format(th_hess))

th_opt = np.linalg.solve(th_hess, -th_grad)
solver.forward(**soltraj_, theta=th_opt)

xss = np.stack(soltraj_["xs"])
times = np.arange(0, T + 1)

plt.figure()
(_lines,) = plt.plot(*xss.T, marker="o", mfc=(0.1, 0.2, 0.9, 0.4))
for i, xp in enumerate(xss):
    label_ = "$x_{{ {:d} }}$".format(i)
    _offs = np.random.rand(2) * 10
    plt.annotate(label_, xp, textcoords="offset points", xytext=_offs, ha="center")

plt.title("$x_t$ (cyclic)")
plt.legend()
plt.tight_layout()
plt.savefig(ASSET_DIR / "gar-cyclic-lqr-2d.png")
plt.savefig(ASSET_DIR / "gar-cyclic-lqr-2d.pdf")

plt.show()
