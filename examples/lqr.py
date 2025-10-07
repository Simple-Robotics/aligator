#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: BSD-2-Clause
# Copyright 2023 Inria

"""Formulating and solving a linear quadratic regulator with Aligator."""

import matplotlib.pyplot as plt
import numpy as np
import tap

import aligator
from aligator import constraints, dynamics, manifolds
from aligator.utils.plotting import plot_convergence


class Args(tap.Tap):
    term_cstr: bool = False
    bounds: bool = False


args = Args().parse_args()
np.random.seed(42)

TAG = "LQR"
if args.bounds:
    TAG += "_bounded"
if args.term_cstr:
    TAG += "_cstr"

nx = 3  # dimension of the state manifold
nu = 3  # dimension of the input
space = manifolds.VectorSpace(nx)
x0 = space.neutral() + (0.2, 0.3, -0.1)

# Linear discrete dynamics: x[t+1] = A x[t] + B u[t] + c
A = np.eye(nx)
A[0, 1] = -0.2
A[1, 0] = 0.2
B = np.eye(nx)[:, :nu]
B[2, :] = 0.4
c = np.zeros(nx)
c[:] = (0.0, 0.0, 0.1)

# Quadratic cost: ½ x^T Q x + ½ u^T R u + x^T N u
Q = 1e-2 * np.eye(nx)
R = 1e-2 * np.eye(nu)
N = 1e-5 * np.eye(nx, nu)

Qf = np.eye(nx)
if args.term_cstr:  # <-- TODO: should it be `not term_cstr`?
    Qf[:, :] = 0.0


# These matrices define the costs and constraints that apply at each stage
# (a.k.a. node) of our trajectory optimization problem
rcost0 = aligator.QuadraticCost(Q, R, N)
term_cost = aligator.QuadraticCost(Qf, R)
dynmodel = dynamics.LinearDiscreteDynamics(A, B, c)
stage = aligator.StageModel(rcost0, dynmodel)
if args.bounds:
    u_min = -0.18 * np.ones(nu)
    u_max = +0.18 * np.ones(nu)
    ctrl_fn = aligator.ControlErrorResidual(nx, np.zeros(nu))
    stage.addConstraint(ctrl_fn, constraints.BoxConstraint(u_min, u_max))

print(stage)

# Build our problem by appending stages and the optional terminal constraint
nsteps = 20
problem = aligator.TrajOptProblem(x0, nu, space, term_cost)

for i in range(nsteps):
    problem.addStage(stage)

xtar2 = 0.1 * np.ones(nx)
if args.term_cstr:
    term_fun = aligator.StateErrorResidual(space, nu, xtar2)
    problem.addTerminalConstraint(term_fun, constraints.EqualityConstraintSet())

# Instantiate a solver separately
mu_init = 2e-3 if args.bounds else 1e-7
verbose = aligator.VerboseLevel.VERBOSE
tol = 1e-8
solver = aligator.SolverProxDDP(tol, mu_init, verbose=verbose)

his_cb = aligator.HistoryCallback(solver)
solver.registerCallback("his", his_cb)
print("Registered callbacks:", solver.getCallbackNames().tolist())
solver.max_iters = 20
solver.rollout_type = aligator.ROLLOUT_LINEAR

u0 = np.zeros(nu)
us_i = [u0] * nsteps
xs_i = aligator.rollout(dynmodel, x0, us_i)
prob_data = aligator.TrajOptData(problem)
problem.evaluate(xs_i, us_i, prob_data)

solver.setup(problem)
solver.run(problem, xs_i, us_i)
res: aligator.Results = solver.results
ws: aligator.Workspace = solver.workspace
print(res)
print(ws.state_dual_infeas.tolist())
print(ws.control_dual_infeas.tolist())

plt.subplot(121)
fig1: plt.Figure = plt.gcf()
fig1.set_figwidth(6.4)
fig1.set_figheight(3.6)

lstyle = {"lw": 0.9, "marker": ".", "markersize": 5}
trange = np.arange(nsteps + 1)
plt.plot(res.xs, ls="-", **lstyle)

if args.term_cstr:
    plt.hlines(
        xtar2,
        *trange[[0, -1]],
        ls="-",
        lw=1.0,
        colors="k",
        alpha=0.4,
        label=r"$x_\mathrm{tar}$",
    )
plt.hlines(
    0.0,
    *trange[[0, -1]],
    ls=":",
    lw=0.6,
    colors="k",
    alpha=0.4,
    label=r"$x=0$",
)
plt.title("State trajectory $x(t)$")
plt.xlabel("Time $i$")
plt.legend(frameon=False)

plt.subplot(122)
plt.plot(res.us, **lstyle)
if args.bounds:
    plt.hlines(
        np.concatenate([u_min, u_max]),
        *trange[[0, -1]],
        ls="-",
        colors="tab:red",
        lw=1.8,
        alpha=0.4,
        label=r"$\bar{u}$",
    )
plt.xlabel("Time $i$")
plt.title("Controls $u(t)$")
plt.legend(frameon=False, loc="lower right")
plt.tight_layout()


fig2: plt.Figure = plt.figure(figsize=(6.4, 3.6))
ax: plt.Axes = fig2.add_subplot()
niter = res.num_iters
ax.hlines(
    tol,
    0,
    niter,
    colors="k",
    linestyles="-",
    linewidth=2.0,
)
plot_convergence(his_cb, ax, res, show_al_iters=True)
ax.set_title("Convergence (constrained LQR)")
fig2.tight_layout()
fig_dicts = {"traj": fig1, "conv": fig2}

for name, _fig in fig_dicts.items():
    _fig: plt.Figure
    _fig.savefig(f"assets/{TAG}_{name}.pdf")

plt.show()
