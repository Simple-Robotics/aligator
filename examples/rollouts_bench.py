import proxddp
from proxddp import manifolds, dynamics, RolloutType

import numpy as np

import example_robot_data as erd

import itertools
import pprint


robot = erd.load("ur5")
rmodel = robot.model
rdata = robot.data

space = manifolds.MultibodyPhaseSpace(rmodel)
tool_id = rmodel.getFrameId("tool0")

dt = 1e-2


def main(roltype, mu_init):
    roltype = RolloutType(roltype)

    nv = rmodel.nv
    nu = nv

    act_matrix = np.eye(nu)
    ode = dynamics.MultibodyFreeFwdDynamics(space, act_matrix)
    dyn_model = dynamics.IntegratorSemiImplEuler(ode, dt)

    Tf = 1.0
    nsteps = int(Tf / dt)

    rcost = proxddp.CostStack(space.ndx, nu)
    w_x = np.eye(space.ndx) * 1e-4
    w_x[nv:] = 1e-2
    w_u = np.eye(nu) * 1e-2
    rcost.addCost(proxddp.QuadraticCost(w_x * dt, w_u * dt))

    w_term_ee = np.eye(3) * 5.0
    p_ref = np.array([0.2, 0.6, 0.4])
    term_cost = proxddp.QuadraticResidualCost(
        proxddp.FrameTranslationResidual(space.ndx, nu, rmodel, p_ref, tool_id),
        w_term_ee,
    )

    stage_model = proxddp.StageModel(rcost, dyn_model)
    stages = [stage_model for _ in range(nsteps)]

    x0 = space.neutral()
    problem = proxddp.TrajOptProblem(x0, stages, term_cost)

    tol = 1e-4
    solver = proxddp.SolverProxDDP(
        tol, mu_init, verbose=proxddp.VerboseLevel.QUIET, max_iters=400
    )
    solver.setup(problem)
    solver.rollout_type = roltype

    xs_init = [x0] * (nsteps + 1)
    us_init = [np.zeros(nu) for _ in range(nsteps)]

    solver.run(problem, xs_init, us_init)

    res = solver.getResults()
    print(roltype, mu_init)
    print(res)
    return res


mu_in_vals = [0.1, 0.01, 0.001, 1e-4, 1e-5]

opts = itertools.product([0, 1], mu_in_vals)
d = {}

for params in opts:
    res = main(*params)
    d[params] = {
        "niters": res.num_iters,
        "p": res.primal_infeas,
        "d": res.dual_infeas,
        "cost": res.traj_cost,
    }


pprint.pprint(d)
