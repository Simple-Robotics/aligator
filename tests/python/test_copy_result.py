#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Test copying solver result object
import aligator
import aligator.dynamics
import aligator.manifolds
import numpy as np
import example_robot_data as erd

import copy

robot = erd.load("ur5")
rmodel = robot.model
rdata = robot.data
space = aligator.manifolds.MultibodyPhaseSpace(rmodel)
actuation_matrix = np.eye(rmodel.nv)

nq = rmodel.nq
nu = rmodel.nv
ndx = 2 * rmodel.nv
dt = 0.01


def get_xs(result):
    return np.stack(result.xs.tolist())


def get_us(result):
    return np.stack(result.us.tolist())


def test_copy():
    np.random.seed(1)
    x0 = space.rand()
    target = space.rand()

    cost = aligator.QuadraticStateCost(space, nu, target, np.eye(ndx) * dt)
    ode_dynamics = aligator.dynamics.MultibodyFreeFwdDynamics(space, actuation_matrix)
    stage_dyn = aligator.dynamics.IntegratorSemiImplEuler(ode_dynamics, dt)
    stage_model = aligator.StageModel(cost, stage_dyn)

    problem = aligator.TrajOptProblem(x0, nu, space, cost)
    nsteps = 20
    for _ in range(nsteps):
        problem.addStage(stage_model)

    tol = 1e-4
    solver = aligator.SolverProxDDP(tol)
    solver.setup(problem)
    solver.max_iters = 3

    xs_init = [space.rand()] * (nsteps + 1)
    us_init = [np.zeros(nu)] * nsteps
    solver.run(problem, xs_init, us_init)
    traj_cost1 = float(solver.results.traj_cost)
    xs1 = get_xs(solver.results)
    us1 = get_us(solver.results)
    result1 = copy.copy(solver.results)

    xs_init = [space.rand()] * (nsteps + 1)
    us_init = [np.zeros(nu)] * nsteps
    solver.run(problem, xs_init, us_init)
    result2 = copy.copy(solver.results)

    # Check the copy preserves the values
    assert result1.traj_cost == traj_cost1
    assert np.allclose(xs1, get_xs(result1))
    assert np.allclose(us1, get_us(result1))

    # Check the second one did changed
    assert result1.traj_cost != result2.traj_cost
    assert not np.allclose(xs1, get_xs(result2))
    assert not np.allclose(us1, get_us(result2))


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main(sys.argv))
