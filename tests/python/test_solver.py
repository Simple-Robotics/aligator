#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 Inria

"""Unit tests for SolverProxDDP."""

import sys

import example_robot_data as erd
from proxddp.manifolds import MultibodyPhaseSpace
import numpy as np
import proxddp
import pytest


def test_no_node():
    robot = erd.load("ur5")
    rmodel = robot.model
    space = MultibodyPhaseSpace(rmodel)
    actuation_matrix = np.eye(rmodel.nv)
    nu = actuation_matrix.shape[1]
    x0 = space.rand()
    mu_init = 4e-2
    rho_init = 1e-2
    verbose = proxddp.VERBOSE
    TOL = 1e-4
    MAX_ITER = 200
    terminal_cost = proxddp.CostStack(space.ndx, nu)
    problem = proxddp.TrajOptProblem(x0, nu, space, terminal_cost)
    solver = proxddp.SolverProxDDP(
        TOL, mu_init, rho_init=rho_init, max_iters=MAX_ITER, verbose=verbose
    )
    solver.setup(problem)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
