"""
Unit tests for SolverProxDDP.
"""

import sys

import numpy as np
import aligator
import pytest

from aligator.dynamics import LinearDiscreteDynamics
from aligator.manifolds import VectorSpace


@pytest.fixture
def lqr_problem():
    nx = 3
    nu = 3
    space = VectorSpace(nx)
    x0 = space.rand()
    A = np.eye(nx)
    A[0, 1] = -0.2
    A[1, 0] = 0.2
    B = np.eye(nx, nu)
    c = np.zeros(nx)
    dyn = LinearDiscreteDynamics(A, B, c)
    Q = 1e-2 * np.eye(nx)
    R = 1e-2 * np.eye(nu)
    N = 1e-5 * np.eye(nx, nu)
    cost = aligator.QuadraticCost(w_x=Q, w_u=R, w_cross=N)
    problem = aligator.TrajOptProblem(x0, nu, space, cost)
    nsteps = 10
    for _ in range(nsteps):
        stage = aligator.StageModel(cost, dyn)
        problem.addStage(stage)
    return problem, nx, nu, x0


def test_fddp_lqr(lqr_problem):
    problem, nx, nu, x0 = lqr_problem
    problem: aligator.TrajOptProblem
    nsteps = problem.num_steps

    tol = 1e-6
    solver = aligator.SolverFDDP(tol, verbose=aligator.VERBOSE)
    solver.setup(problem)
    solver.max_iters = 2
    xs_init = [x0] * (nsteps + 1)
    us_init = [np.zeros(nu)] * nsteps
    conv = solver.run(problem, xs_init, us_init)
    assert conv


@pytest.mark.skipif(
    not aligator.has_pinocchio_features(),
    reason="Aligator was compiled without Pinocchio features.",
)
def test_no_node():
    import example_robot_data as erd
    from aligator.manifolds import MultibodyPhaseSpace

    robot = erd.load("ur5")
    rmodel = robot.model
    space = MultibodyPhaseSpace(rmodel)
    actuation_matrix = np.eye(rmodel.nv)
    nu = actuation_matrix.shape[1]
    x0 = space.rand()
    mu_init = 4e-2
    TOL = 1e-4
    MAX_ITER = 200
    terminal_cost = aligator.CostStack(space, nu)
    problem = aligator.TrajOptProblem(x0, nu, space, terminal_cost)
    solver = aligator.SolverProxDDP(
        TOL, mu_init, max_iters=MAX_ITER, verbose=aligator.VERBOSE
    )
    solver.setup(problem)


@pytest.fixture
def lqr_problem_constrained(lqr_problem):
    problem, nx, nu, x0 = lqr_problem
    ctrl_fn = aligator.ControlErrorResidual(nx, np.zeros(nu))
    for stage in problem.stages:
        stage: aligator.StageModel
        umin = np.array([-1, -1, -1])
        umax = np.array([1, 1, 1])
        stage.addConstraint(ctrl_fn, aligator.constraints.BoxConstraint(umin, umax))
        space = stage.xspace
    xf = x0 * 0.9
    state_fn = aligator.StateErrorResidual(space, nu, xf)
    problem.addTerminalConstraint(
        state_fn, aligator.constraints.EqualityConstraintSet()
    )
    return lqr_problem


@pytest.mark.parametrize(
    "strategy",
    [
        aligator.SA_FILTER,
        aligator.SA_LINESEARCH_ARMIJO,
        aligator.SA_LINESEARCH_NONMONOTONE,
    ],
)
def test_proxddp_lqr(lqr_problem_constrained, strategy):
    problem, nx, nu, x0 = lqr_problem_constrained
    problem: aligator.TrajOptProblem
    nsteps = problem.num_steps

    tol = 1e-6
    mu_init = 1e-2
    solver = aligator.SolverProxDDP(tol, mu_init, verbose=aligator.VERBOSE)
    solver.setup(problem)
    solver.sa_strategy = strategy
    solver.max_iters = 4
    xs_init = [np.random.randn(nx)] * (nsteps + 1)
    xs_init[0] = x0
    us_init = [np.zeros(nu)] * nsteps
    assert solver.run(problem, xs_init, us_init)
    print(solver.results)
    print(solver.mu)
    res: aligator.Results = solver.results
    res_copy: aligator.Results = res.copy()

    assert res_copy.traj_cost == res.traj_cost
    assert np.allclose(np.concatenate(res.xs), np.concatenate(res_copy.xs))
    assert np.allclose(np.concatenate(res.us), np.concatenate(res_copy.us))
    assert np.allclose(np.concatenate(res.vs), np.concatenate(res_copy.vs))
    assert np.allclose(np.concatenate(res.lams), np.concatenate(res_copy.lams))


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
