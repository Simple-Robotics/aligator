"""
Tests for GAR module in the aligator Python bindings.
"""

from aligator import gar
import numpy as np
import pprint

import matplotlib.pyplot as plt
import tap
import eigenpy

import utils  # noqa

np.set_printoptions(precision=5, linewidth=250)


class Args(tap.Tap):
    term: bool = False
    mid: bool = False


args = Args().parse_args()
nx = 1
nu = 1
x0 = 1.0 * np.ones(nx)
xterm = np.array([0.0])


def knot_get_default(nx, nu, nc):
    knot = gar.LQRKnot(nx, nu, nc)
    knot.Q[:] = np.eye(nx, nx) * 0.1
    knot.q[:] = 0.0
    knot.R[:] = np.eye(nu) * 0.1
    knot.A[:] = 1.2 * np.eye(nx)
    knot.f[:] = 0.01 * np.ones(nx)
    knot.B = np.eye(nx, nu)
    knot.E[:] = -np.eye(nx)
    return knot


knot_base = knot_get_default(nx, nu, 0)

print(f"xf = {xterm}")
print(f"x0 = {x0}")


# terminal knot
if args.term:
    knot1 = knot_get_default(nx, 0, nx)
    knot1.C = np.eye(nx, nx)
    knot1.d = -xterm
else:
    knot1 = knot_get_default(nx, 0, 0)
    knot1.Q[:] = np.eye(nx) * 0.1
    knot1.q[:] = -knot1.Q @ xterm

T = 7
t0 = T // 2


def make_interm_node(t0):
    kn = knot_get_default(nx, nu, nc=nx)
    xmid = np.array([-0.3])
    kn.C = np.eye(nx, nx)
    kn.d = -xmid
    return kn


knots = [knot_base]
for t in range(T - 1):
    if args.mid and t == t0:
        knot_mid = make_interm_node(t)
    else:
        knot_mid = knot_base.copy()
    knots.append(knot_mid)
knots.append(knot1)
# constructor creates a copy
prob = gar.LQRProblem(knots, nx)
prob.G0 = -np.eye(nx)
prob.g0 = x0

print("Is problem parameterized? {}".format(prob.isParameterized))
ricsolve = gar.ProximalRiccatiSolver(prob)

assert prob.horizon == T
mu = 1e-5
mueq = mu
ricsolve.backward(mu, mueq)


def inftyNorm(x):
    return np.max(np.abs(x))


def get_np_solution():
    matrix, rhs = gar.lqrDenseMatrix(prob, mu, mueq)
    knots = prob.stages
    ldlt = eigenpy.LDLT(matrix)
    _sol_np = ldlt.solve(-rhs)
    err = rhs + matrix @ _sol_np
    print("LDLT solve err.: {:4.3e}".format(inftyNorm(err)))
    xs = []
    us = []
    vs = []
    nc0 = prob.g0.size
    lbdas = [_sol_np[:nc0]]

    i = prob.g0.size
    for t in range(T + 1):
        knot = knots[t]
        nx = knot.nx
        nu = knot.nu
        nc = knot.nc
        n = nx + nu + nc
        blk = _sol_np[i : i + n + nx]
        xs.append(blk[:nx])
        us.append(blk[nx : nx + nu])
        vs.append(blk[nx + nu : n])
        if t < T:
            lbdas.append(blk[n : n + nx])
        i += n + nx

    out = dict(xs=xs, us=us, vs=vs, lbdas=lbdas)
    print("=======")
    return out


sol_dense = get_np_solution()

sol_gar = gar.lqrInitializeSolution(prob)
xs_out = sol_gar["xs"]
us_out = sol_gar["us"]
vs_out = sol_gar["vs"]
lbdas_out = sol_gar["lbdas"]

ricsolve.forward(**sol_gar)


def checkAllErrors(sol: dict, knots):
    xs = sol["xs"]
    us = sol["us"]
    vs = sol["vs"]
    lbdas = sol["lbdas"]
    _ds = []
    for t in range(T):
        x = xs[t]
        u = us[t]
        v = vs[t]
        lbda = lbdas[t + 1]
        xn = xs[t + 1]
        knot = knots[t]
        rdl = knot.E @ xn + knot.A @ x + knot.B @ u + knot.f - mu * lbda
        gu = knot.S.T @ x + knot.R @ u + knot.D.T @ v + knot.B.T @ lbda + knot.r
        if knot.nc > 0:
            gc = knot.C @ x + knot.D @ u + knot.d - mueq * v
        else:
            gc = 0.0
        d = {
            "dyn": inftyNorm(rdl),
            "u": inftyNorm(gu),
            "c": inftyNorm(gc),
            "nc": knot.nc,
        }
        gx = knot.q + knot.Q @ x + knot.S @ u + knot.A.T @ lbda + knot.C.T @ v
        if t > 0:
            gx += knots[t - 1].E.T @ lbdas[t]
        d["x"] = inftyNorm(gx)
        _ds.append(d)
    pprint.pp(_ds)


print("dense:")
checkAllErrors(sol_dense, prob.stages)
print("gar:")
checkAllErrors(sol_gar, prob.stages)

print("Optimal value (dense):", prob.evaluate(sol_gar["xs"], sol_gar["us"]))
print("Optimal value (gar):  ", prob.evaluate(sol_dense["xs"], sol_dense["us"]))

# Plot solution

plt.figure(figsize=(8.0, 4.0))
plt.subplot(121)
xss = np.stack(xs_out)
xsd = np.stack(sol_dense["xs"])
times = np.arange(T + 1)
i = 0
plt.plot(times, xss[:, i], marker=".", ls="--", markersize=10, alpha=0.7, label="gar")
plt.plot(times, xsd[:, i], marker=".", ls="--", markersize=10, alpha=0.7, label="dense")
plt.scatter(times[0], x0[i], c="cyan", s=14, zorder=2, label="$x_0$")
plt.scatter(times[T], xterm[i], c="r", s=14, zorder=2, label="$x_\\mathrm{f}$")
plt.grid(True)
plt.legend()
plt.title("State $x_t$")

plt.subplot(122)

lss = np.stack(lbdas_out)
lsd = np.stack(sol_dense["lbdas"])
plt.plot(times, lss[:, i], marker=".", ls="--", markersize=10, alpha=0.7, label="gar")
plt.plot(times, lsd[:, i], marker=".", ls="--", markersize=10, alpha=0.7, label="dense")
plt.grid(True)
plt.legend()
plt.title("Co-state $\\lambda_t$")
plt.tight_layout()

plt.show()


PARAM_DIM = nx
prob.addParameterization(PARAM_DIM)

knot1 = prob.stages[-1]
knot1.Gx[:] = -knot1.Q
knot1.Gth[:] = knot1.Q
print("Terminal knot:", knot1)

print("Is problem parameterized? {}".format(prob.isParameterized))
ricsolve = gar.ProximalRiccatiSolver(prob)
ricsolve.backward(mu, mueq)

vm0: gar.value_data = ricsolve.datas[0].vm
thGrad = ricsolve.thGrad
thHess = ricsolve.thHess
print("grad:", thGrad[0])
print("hess:", thHess[0, 0])

ths_ = []
vths_calc_ = []


def parametric_solve(v):
    import copy

    param_sol = copy.deepcopy(sol_gar)
    _theta = np.array([v])
    ths_.append(_theta)
    ricsolve.forward(theta=_theta, **param_sol)

    xss = np.stack(param_sol["xs"])
    plt.plot(times, xss[:, 0], marker=".", lw=1.0, label="{:.2e}".format(_theta[0]))

    vths_calc_.append(prob.evaluate(param_sol["xs"], param_sol["us"], theta=_theta))


plt.figure(figsize=(9.6, 4.8))
plt.subplot(121)
iids = np.linspace(-1.0, 1.0, 17, endpoint=True)
for v in iids:
    parametric_solve(v)
plt.hlines(xterm[0], times[0], times[-1], colors="r", linestyles="-", lw=1.4, zorder=2)
plt.title("State $x_t$")
plt.xlabel("Discrete time $t$")
plt.grid(True)
plt.legend(title="$\\theta$", fontsize="x-small", ncols=2)

plt.subplot(122)
plt.title("Value curve $V^\\star(\\theta)$")
ths_ = np.array(ths_)
vths_calc_ = np.array(vths_calc_)
vths_pred_ = [t.dot(thGrad) + 0.5 * t.dot(thHess @ t) for t in ths_]
_dv = vths_calc_[0] - vths_pred_[0]
vths_pred_ += _dv

plt.hlines(_dv, ths_.min(), ths_.max(), ls="--", colors="k")
plt.plot(ths_, vths_pred_, marker="+", alpha=0.6, label="predicted")
plt.plot(ths_, vths_calc_, marker=".", alpha=0.6, label="evaluate (forward)")
plt.legend()
plt.xlabel("$\\theta$")
plt.tight_layout()

plt.show()
