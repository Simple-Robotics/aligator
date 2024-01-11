#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 Inria

"""Unit tests for SolverProxDDP."""

import sys

import example_robot_data as erd
from aligator.manifolds import VectorSpace, MultibodyPhaseSpace
import numpy as np
import aligator
import pytest


def test_fddp_lqr():
    nx = 3
    nu = 2
    space = VectorSpace(nx)
    x0 = space.rand()
    A = np.eye(nx)
    B = np.ones((nx, nu))
    c = np.zeros(nx)
    dyn = aligator.dynamics.LinearDiscreteDynamics(A, B, c)
    w_x = np.eye(nx)
    w_u = np.eye(nu)
    cost = aligator.QuadraticCost(w_x, w_u)
    problem = aligator.TrajOptProblem(x0, nu, space, cost)
    nsteps = 10
    for i in range(nsteps):
        stage = aligator.StageModel(cost, dyn)
        problem.addStage(stage)

    tol = 1e-6
    solver = aligator.SolverFDDP(tol, aligator.VerboseLevel.VERBOSE)
    solver.setup(problem)
    solver.max_iters = 2
    xs_init = [x0] * (nsteps + 1)
    us_init = [np.zeros(nu)] * nsteps
    conv = solver.run(problem, xs_init, us_init)
    assert conv


def test_no_node():
    robot = erd.load("ur5")
    rmodel = robot.model
    space = MultibodyPhaseSpace(rmodel)
    actuation_matrix = np.eye(rmodel.nv)
    nu = actuation_matrix.shape[1]
    x0 = space.rand()
    mu_init = 4e-2
    rho_init = 1e-2
    verbose = aligator.VERBOSE
    TOL = 1e-4
    MAX_ITER = 200
    terminal_cost = aligator.CostStack(space, nu)
    problem = aligator.TrajOptProblem(x0, nu, space, terminal_cost)
    solver = aligator.SolverProxDDP(
        TOL, mu_init, rho_init=rho_init, max_iters=MAX_ITER, verbose=verbose
    )
    solver.setup(problem)


def test_proxddp_lqr():
    nx = 3
    nu = 3
    space = VectorSpace(nx)
    x0 = space.neutral() + (0.2, 0.3, -0.1)
    xf = x0 * 0.9
    umin = np.array([-1, -1])
    umax = np.array([1, 1])
    A = np.eye(nx)
    A[0, 1] = -0.2
    A[1, 0] = 0.2
    B = np.eye(nx)[:, :nu]
    B[2, :] = 0.4
    c = np.zeros(nx)
    c[:] = (0.0, 0.0, 0.1)

    Q = 1e-2 * np.eye(nx)
    R = 1e-2 * np.eye(nu)
    N = 1e-5 * np.eye(nx, nu)

    run_cost = aligator.QuadraticCost(Q, R, N)
    term_cost = aligator.QuadraticCost(Q, R)
    dyn = aligator.dynamics.LinearDiscreteDynamics(A, B, c)
    ctrl_fn = aligator.ControlErrorResidual(space.ndx, np.zeros(nu))
    state_fn = aligator.StateErrorResidual(space, nu, xf)
    stage = aligator.StageModel(run_cost, dyn)
    stage.addConstraint(ctrl_fn, aligator.constraints.BoxConstraint(umin, umax))

    nsteps = 20
    stages = [stage] * nsteps
    problem = aligator.TrajOptProblem(x0, stages, term_cost)

    term_cstr = aligator.StageConstraint(
        state_fn, aligator.constraints.EqualityConstraintSet()
    )
    problem.addTerminalConstraint(term_cstr)

    tol = 1e-6
    mu_init = 1e-3
    rho_init = 1e-2
    solver = aligator.SolverProxDDP(
        tol, mu_init, rho_init, aligator.VerboseLevel.VERBOSE
    )
    solver.setup(problem)
    solver.sa_strategy = aligator.SA_FILTER
    solver.max_iters = 50
    xs_init = [x0] * (nsteps + 1)
    us_init = [np.zeros(nu)] * nsteps
    conv = solver.run(problem, xs_init, us_init)
    assert conv


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
