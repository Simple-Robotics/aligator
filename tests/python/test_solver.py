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


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
