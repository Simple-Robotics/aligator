#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: BSD-2-Clause
# Copyright 2023 Inria

"""Formulating and solving a linear quadratic regulator with Aligator."""

from copy import deepcopy

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


# Build our problem by appending stages and the optional terminal constraint
nsteps = 20
problem = aligator.TrajOptProblem(x0, nu, space, term_cost)

for i in range(nsteps):
    problem.addStage(stage)

xtar2 = 0.1 * np.ones(nx)
if args.term_cstr:
    term_fun = aligator.StateErrorResidual(space, nu, xtar2)
    problem.addTerminalConstraint(
        aligator.StageConstraint(term_fun, constraints.EqualityConstraintSet())
    )

# Instantiate a solver separately
mu_init = 1e-3 if args.bounds else 1e-6
rho_init = 0.0
verbose = aligator.VerboseLevel.VERBOSE
tol = 1e-8
solver = aligator.SolverProxDDP(tol, mu_init, rho_init, verbose=verbose)


class CustomCallback(aligator.BaseCallback):
    def __init__(self):
        super().__init__()
        self.active_sets = []
        self.x_dirs = []
        self.u_dirs = []
        self.lams = []
        self.Qus = []
        self.kkts = []

    def call(self, workspace: aligator.Workspace, results: aligator.Results):
        self.active_sets.append(workspace.active_constraints.tolist())
        self.x_dirs.append(deepcopy(workspace.dxs.tolist()))
        self.u_dirs.append(deepcopy(workspace.dus.tolist()))
        self.lams.append(deepcopy(results.lams.tolist()))
        Qus = [qq.Qu.copy() for qq in workspace.q_params]
        self.Qus.append(Qus)
        kkts = workspace.kkt_mat
        self.kkts.append(deepcopy(kkts))


cus_cb = CustomCallback()
solver.registerCallback("cus", cus_cb)
his_cb = aligator.HistoryCallback()
solver.registerCallback("his", his_cb)
solver.max_iters = 20

u0 = np.zeros(nu)
us_i = [u0] * nsteps
xs_i = aligator.rollout(dynmodel, x0, us_i)
prob_data = aligator.TrajOptData(problem)
problem.evaluate(xs_i, us_i, prob_data)

solver.setup(problem)
for i in range(nsteps):
    psc = solver.workspace.getConstraintScaler(i)
    if args.bounds:
        psc.set_weight(100, 1)
solver.run(problem, xs_i, us_i)
res = solver.results
ws = solver.workspace

plt.subplot(121)
fig1: plt.Figure = plt.gcf()

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


fig2: plt.Figure = plt.figure()
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
plot_convergence(his_cb, ax, res)
ax.set_title("Convergence (constrained LQR)")
ax.legend(
    [
        "Tolerance $\\epsilon_\\mathrm{tol}$",
        "Primal error $p$",
        "Dual error $d$",
    ]
)
fig2.tight_layout()

plt.show()
