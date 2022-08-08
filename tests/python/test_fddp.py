import proxddp
import numpy as np

from proxddp import SolverFDDP
import pytest


def test_fddp_lqr():
    nx = 3
    nu = 2
    space = proxddp.manifolds.VectorSpace(nx)
    x0 = space.rand()
    A = np.eye(nx)
    B = np.ones((nx, nu))
    c = np.zeros(nx)
    dyn = proxddp.dynamics.LinearDiscreteDynamics(A, B, c)
    w_x = np.eye(nx)
    w_u = np.eye(nu)
    cost = proxddp.QuadraticCost(w_x, w_u)
    problem = proxddp.TrajOptProblem(x0, nu, space, cost)
    nsteps = 10
    for i in range(nsteps):
        stage = proxddp.StageModel(space, nu, cost, dyn)
        problem.addStage(stage)

    tol = 1e-6
    solver = SolverFDDP(tol, 1e-10, proxddp.VerboseLevel.VERBOSE)
    solver.setup(problem)
    solver.max_iters = 1
    xs_init = [x0] * (nsteps + 1)
    us_init = [np.zeros(nu)] * nsteps
    conv = solver.run(problem, xs_init, us_init)
    assert conv


if __name__ == "__main__":
    import sys

    retcode = pytest.main(sys.argv)
