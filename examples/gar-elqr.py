"""
Tests for GAR module in the proxddp Python bindings.
"""
from proxddp import gar
import numpy as np
import pprint

import matplotlib.pyplot as plt
import tap
import eigenpy

np.random.seed(42)
np.set_printoptions(precision=3, linewidth=250)


class Args(tap.Tap):
    term: bool = False
    mid: bool = False


args = Args().parse_args()
nx = 1
nu = 1
x0 = 1.0 * np.ones(nx)
xbar = np.ones(nx) * 0.2
xterm = np.array([-1.5])


def knot_get_default(nx, nu, nc):
    knot = gar.LQRKnot(nx, nu, nc)
    knot.Q[:] = np.eye(nx, nx) * 0.1
    knot.q[:] = -knot.Q @ xbar
    knot.R[:] = np.eye(nu) * 0.1
    knot.A[:] = 1.2 * np.eye(nx)
    knot.B = np.eye(nx, nu)
    knot.E[:] = -np.eye(nx)
    return knot


knot_base = knot_get_default(nx, nu, 0)

knot0 = knot_get_default(nx, nu, nx)
knot0.C[:] = -np.eye(nx, nx)
knot0.d = x0

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
    knot1.q = -knot1.Q @ xterm

T = 20
t0 = T // 2


def make_interm_node(t0):
    kn = knot_get_default(nx, nu, nc=nx)
    xmid = np.array([-0.3])
    kn.C = np.eye(nx, nx)
    kn.d = -xmid
    return kn


knots = [knot0]
for t in range(T - 1):
    if args.mid and t == t0:
        kn = make_interm_node(t)
    else:
        kn = knot_base.copy()
    knots.append(kn)
knots.append(knot1)

# print(f"{knots[0]}")
# print(f"{knots[1]}")

ricsolve = gar.ProximalRiccatiBwd(knots)

assert ricsolve.horizon == T
mu = 1e-5
# mueq = 0.01
mueq = mu
ricsolve.backward(mu, mueq)


def inftyNorm(x):
    return np.max(np.abs(x))


def get_np_solution():
    matrix, rhs = gar.lqrDenseMatrix(knots, mu, mueq)
    print(f"Problem matrix:\n{matrix}")
    print(f"Problem rhs:\n{rhs}")

    ldlt = eigenpy.LDLT(matrix)
    _sol_np = ldlt.solve(-rhs)
    err = rhs + matrix @ _sol_np
    print("LDLT solve err.: {:4.3e}".format(inftyNorm(err)))
    xs = []
    us = []
    vs = []
    lbdas = []
    i = 0
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
    pprint.pp(out, indent=2)
    print("=======")
    return out


sol_dense = get_np_solution()

xs_out = [np.zeros(nx) for _ in range(T + 1)]
us_out = [np.zeros(nu) for _ in range(T)]
vs_out = [np.zeros(knot.nc) for knot in knots]
lbdas_out = [np.zeros(knots[t].nx) for t in range(T)]
sol_gar = {"xs": xs_out, "us": us_out, "vs": vs_out, "lbdas": lbdas_out}

ricsolve.forward(**sol_gar)

pprint.pp(sol_gar, indent=2)


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
        lbda = lbdas[t]
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
            gx += knots[t - 1].E.T @ lbdas[t - 1]
        d["x"] = inftyNorm(gx)
        _ds.append(d)
    pprint.pp(_ds)


print("dense:")
checkAllErrors(sol_dense, knots)
print("gar:")
checkAllErrors(sol_gar, knots)

# Plot solution

plt.figure()
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

plt.show()
