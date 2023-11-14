from aligator import gar

import numpy as np

import matplotlib.pyplot as plt
from utils import ASSET_DIR


np.random.seed(42)
np.set_printoptions(precision=3, linewidth=250)

nx = 1
nu = 1


def create_knot(nx, nu):
    knot = gar.LQRKnot(nx, nu, 0)
    knot.Q[:] = np.eye(nx) * 0.008
    knot.R[:] = np.eye(nu) * 0.01
    knot.A[:] = 1.2
    knot.B[:] = np.eye(nx, nu)
    knot.E[:] = -np.eye(nx)
    return knot


T = 30
knots = [create_knot(nx, nu) for _ in range(T)]
knots.append(create_knot(nx, 0))
prob = gar.LQRProblem(knots, 0)

PARAM_DIM = nx
prob.addParameterization(PARAM_DIM)

xf = np.array([0.05])
kf = prob.stages[T]
kf.Q[:] = np.eye(nx) * 1.0
kf.q[:] = -kf.Q @ xf


def add_mid(t0, v):
    kt0 = prob.stages[t0]
    kt0.Q[:] = 0.05
    kt0.q[:] = -kt0.Q @ np.array([v])


t0 = T // 3
add_mid(t0, 0.1)
t1 = 2 * T // 3
add_mid(t1, -0.1)

print(prob.stages[0])
print(prob.stages[T])
prob.stages[0].Gammax[:] = +np.eye(nx)
prob.stages[T].Gammax[:] = -np.eye(nx)

solver = gar.ProximalRiccatiSolver(prob)
mu = 1e-8
solver.backward(mu, mu)

sol_ = gar.lqrInitializeSolution(prob)


th_grad = solver.thGrad
th_hess = solver.thHess
print("thGrad:", th_grad)
print("thHess:", th_hess)

th_opt = np.linalg.solve(th_hess, -th_grad)
solver.forward(**sol_, theta=th_opt)

xss = np.stack(sol_["xs"])
times = np.arange(0, T + 1)

print("xs[0] = ", xss[0])
plt.figure()
plt.plot(times, xss, marker="x")
plt.hlines(
    xss[0],
    times[0],
    times[-1],
    colors="k",
    linestyles="--",
    label="$x^\\star_0 = x^\\star_T$",
)
ax = plt.gca()
ax.xaxis.set_major_locator(plt.MultipleLocator(2))

plt.title("$x_t$ (cyclic)")
plt.legend()
plt.tight_layout()
plt.savefig(ASSET_DIR / "gar-cyclic-lqr-1d.png")

plt.show()
