"""
Test MPC (cycling OCP)
"""

import pinocchio as pin
import numpy as np
import aligator
import pytest


NUM_MPC_CYCLES = 100


@pytest.fixture
def mpc_problem():
    rmodel = pin.buildSampleModelHumanoid()
    rdata: pin.Data = rmodel.createData()
    space = aligator.manifolds.MultibodyPhaseSpace(rmodel)

    dt = 1e-2  # Timestep

    # Contact models
    FOOT_FRAME_IDS = {
        fname: rmodel.getFrameId(fname)
        for fname in ["lleg_effector_body", "rleg_effector_body"]
    }
    FOOT_JOINT_IDS = {
        fname: rmodel.frames[fid].parentJoint for fname, fid in FOOT_FRAME_IDS.items()
    }

    nv = rmodel.nv
    nu = rmodel.nv
    act_matrix = np.eye(nv, nu, -6)
    prox_settings = pin.ProximalSettings(1e-9, 1e-10, 10)
    constraint_models = []
    constraint_datas = []
    for fname, fid in FOOT_FRAME_IDS.items():
        joint_id = FOOT_JOINT_IDS[fname]
        pl1 = rmodel.frames[fid].placement
        pl2 = rdata.oMf[fid]
        cm = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_3D,
            rmodel,
            joint_id,
            pl1,
            0,
            pl2,
            pin.LOCAL,
        )
        cm.corrector.Kp[:] = (0, 0, 0)
        cm.corrector.Kd[:] = (0, 0, 0)
        constraint_models.append(cm)
        constraint_datas.append(cm.createData())

    q0 = space.neutral()
    x0 = np.concatenate((q0, np.zeros(nv)))
    u0 = np.zeros(nu)

    T = 50

    stages = []
    for _ in range(T):
        rcost = aligator.CostStack(space, nu)
        rcost.addCost(aligator.QuadraticStateCost(space, nu, x0, np.eye(space.ndx)))
        rcost.addCost(aligator.QuadraticControlCost(space, u0, np.eye(nu) * 1e-4))

        ode = aligator.dynamics.MultibodyConstraintFwdDynamics(
            space, act_matrix, constraint_models, prox_settings
        )
        dyn_model = aligator.dynamics.IntegratorSemiImplEuler(ode, dt)
        stm = aligator.StageModel(rcost, dyn_model)
        stages.append(stm)

    term_cost = aligator.CostStack(space, nu)
    problem = aligator.TrajOptProblem(x0, stages, term_cost)
    return problem


def test_parallel_mpc(mpc_problem):
    problem = mpc_problem

    TOL = 1e-5
    mu_init = 1e-8
    verbose = aligator.VerboseLevel.QUIET
    solver = aligator.SolverProxDDP(TOL, mu_init, verbose=verbose)
    solver.rollout_type = aligator.ROLLOUT_LINEAR
    solver.max_iters = 100
    solver.sa_strategy = aligator.SA_FILTER
    solver.force_initial_condition = True
    solver.linear_solver_choice = aligator.LQ_SOLVER_PARALLEL
    solver.setNumThreads(2)
    solver.filter.beta = 1e-5
    solver.setup(problem)

    solver.run(problem, [], [])

    xs = solver.results.xs.tolist().copy()
    us = solver.results.us.tolist().copy()

    # Launch MPC
    for t in range(NUM_MPC_CYCLES):
        print("Time " + str(t))

        xs = xs[1:] + [xs[-1]]
        us = us[1:] + [us[-1]]

        problem.x0_init = xs[0]
        solver.setup(problem)
        solver.run(problem, xs, us)

        xs = solver.results.xs.tolist().copy()
        us = solver.results.us.tolist().copy()


def test_serial_mpc(mpc_problem):
    problem = mpc_problem

    TOL = 1e-5
    mu_init = 1e-8
    verbose = aligator.VerboseLevel.QUIET
    solver = aligator.SolverProxDDP(TOL, mu_init, verbose=verbose)
    solver.rollout_type = aligator.ROLLOUT_LINEAR
    solver.max_iters = 100
    solver.sa_strategy = aligator.SA_FILTER
    solver.force_initial_condition = True
    solver.linear_solver_choice = aligator.LQ_SOLVER_SERIAL
    solver.filter.beta = 1e-5
    solver.setup(problem)

    solver.run(problem, [], [])

    xs = solver.results.xs.tolist().copy()
    us = solver.results.us.tolist().copy()

    # Launch MPC
    for t in range(NUM_MPC_CYCLES):
        print("Time " + str(t))

        xs = xs[1:] + [xs[-1]]
        us = us[1:] + [us[-1]]

        problem.x0_init = xs[0]
        solver.setup(problem)
        solver.run(problem, xs, us)

        xs = solver.results.xs.tolist().copy()
        us = solver.results.us.tolist().copy()


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main(sys.argv))
