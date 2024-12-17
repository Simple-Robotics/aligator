from aligator import gar

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
    knot = gar.LqrKnot(nx, nu, 0)
    knot.Q[:] = np.eye(nx) * 1e-3
    knot.R[:] = np.eye(nu) * 0.1
    th = 0.156
    cs = np.cos(th)
    ss = np.sin(th)
    knot.A[:] = [[cs, -ss], [ss, cs]]
    knot.B[:] = np.eye(nx, nu)
    knot.E[:] = -np.eye(nx)
    knot.f[:] = 0.0
    return knot


T = 20
base_knot = create_knot(nx, nu)
knots = [base_knot for _ in range(T)]
knots.append(create_knot(nx, 0))
prob = gar.LqrProblem(knots, 0)

PARAM_DIM = nx
prob.addParameterization(PARAM_DIM)

xf = np.array([0.6, 0.6])
kf = prob.stages[T]
kf.Q[:] = np.eye(nx) * 1.0
kf.q[:] = -kf.Q @ xf


_t_objs = []
_x_objs = []


def add_mid(t0, v):
    kt0 = prob.stages[t0]
    kt0.Q[:] = np.eye(nx) * 0.2
    kt0.q[:] = -kt0.Q @ v
    _t_objs.append(t0)
    _x_objs.append(np.array(v))


t0 = T // 4
xt0 = np.array([-1.0, 0.2])
add_mid(t0, xt0)
t1 = T // 2
xt1 = (1.0, 1.0)
# add_mid(t1, xt1)
xt2 = (1.4, -1.4)
t2 = 3 * T // 4
add_mid(t2, xt2)

print(prob.stages[0])
print(prob.stages[T])
prob.stages[0].Gx[:] = +np.eye(nx)
prob.stages[T].Gx[:] = -np.eye(nx)

solver = gar.ProximalRiccatiSolver(prob)
mu = 1e-12
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

plt.figure(figsize=(5.4, 4))
(_lines,) = plt.plot(*xss.T, zorder=-1)
_colors = np.ones((T + 1, 4))
_colors[:T] = (0.1, 0.2, 0.9, 0.4)
_colors[T] = (0.0, 0.8, 0.1, 1.0)
plt.scatter(*xss.T, marker="o", s=20, c=_colors)
# import IPython; IPython.embed()
for i, xp in enumerate(xss):
    if i not in [0, 1, 2, T - 1] + _t_objs:
        continue
    label_ = "$x_{{ {:d} }}$".format(i)
    _offs = (10, 3)
    plt.annotate(label_, xp, textcoords="offset points", xytext=_offs, ha="center")

plt.title("Cylic LQ problem (2D)\nConstraint $x_0=x_{{{}}}$".format(T))
plt.scatter(0.0, 0.0, marker=".", c="k")
plt.annotate("$0$", (0, 0), xytext=(0.03, 0.03))
for i, _x in enumerate(_x_objs):
    plt.scatter(*_x, marker="o", c="red")
    plt.annotate("$\\bar{{x}}_{{{}}}$".format(_t_objs[i]), _x, xytext=_x - (0.18, 0.03))

plt.axis("equal")
plt.tight_layout()
plt.savefig(ASSET_DIR / "gar-cyclic-lqr-2d.png")
plt.savefig(ASSET_DIR / "gar-cyclic-lqr-2d.pdf")

plt.show()
