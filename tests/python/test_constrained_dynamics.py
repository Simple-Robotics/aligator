import pytest

import pinocchio as pin
import numpy as np
import hppfcl as fcl
import aligator

from aligator import manifolds, dynamics
from utils import finite_diff

TOL = 1e-4


def createFourBarLinkages():
    # adapted from pinocchio/examples/simulation-closed-kinematic-chains.py
    height = 0.1
    width = 0.01
    radius = 0.05

    mass_link_A = 10.0
    length_link_A = 1.0
    shape_link_A = fcl.Capsule(radius, length_link_A)

    mass_link_B = 5.0
    length_link_B = 0.6
    shape_link_B = fcl.Capsule(radius, length_link_B)

    inertia_link_A = pin.Inertia.FromBox(mass_link_A, length_link_A, width, height)
    placement_center_link_A = pin.SE3.Identity()
    placement_center_link_A.translation = pin.XAxis * length_link_A / 2.0
    placement_shape_A = placement_center_link_A.copy()
    placement_shape_A.rotation = pin.Quaternion.FromTwoVectors(
        pin.ZAxis, pin.XAxis
    ).matrix()

    inertia_link_B = pin.Inertia.FromBox(mass_link_B, length_link_B, width, height)
    placement_center_link_B = pin.SE3.Identity()
    placement_center_link_B.translation = pin.XAxis * length_link_B / 2.0
    placement_shape_B = placement_center_link_B.copy()
    placement_shape_B.rotation = pin.Quaternion.FromTwoVectors(
        pin.ZAxis, pin.XAxis
    ).matrix()

    model = pin.Model()
    collision_model = pin.GeometryModel()

    RED_COLOR = np.array([1.0, 0.0, 0.0, 1.0])
    WHITE_COLOR = np.array([1.0, 1.0, 1.0, 1.0])
    TRANS_COLOR = np.array([1.0, 1.0, 1.0, 0.5])

    base_joint_id = 0
    geom_obj0 = pin.GeometryObject(
        "link_A1",
        base_joint_id,
        shape_link_A,
        pin.SE3(
            pin.Quaternion.FromTwoVectors(pin.ZAxis, pin.XAxis).matrix(),
            np.zeros(3),
        ),
    )
    geom_obj0.meshColor = TRANS_COLOR
    collision_model.addGeometryObject(geom_obj0)

    joint1_placement = pin.SE3.Identity()
    joint1_placement.translation = pin.XAxis * length_link_A / 2.0
    joint1_id = model.addJoint(
        base_joint_id, pin.JointModelRY(), joint1_placement, "link_B1"
    )
    model.appendBodyToJoint(joint1_id, inertia_link_B, placement_center_link_B)
    geom_obj1 = pin.GeometryObject(
        "link_B1", joint1_id, shape_link_B, placement_shape_B
    )
    geom_obj1.meshColor = RED_COLOR
    collision_model.addGeometryObject(geom_obj1)

    joint2_placement = pin.SE3.Identity()
    joint2_placement.translation = pin.XAxis * length_link_B
    joint2_id = model.addJoint(
        joint1_id, pin.JointModelRY(), joint2_placement, "link_A2"
    )
    model.appendBodyToJoint(joint2_id, inertia_link_A, placement_center_link_A)
    geom_obj2 = pin.GeometryObject(
        "link_A2", joint2_id, shape_link_A, placement_shape_A
    )
    geom_obj2.meshColor = WHITE_COLOR
    collision_model.addGeometryObject(geom_obj2)

    joint3_placement = pin.SE3.Identity()
    joint3_placement.translation = pin.XAxis * length_link_A
    joint3_id = model.addJoint(
        joint2_id, pin.JointModelRY(), joint3_placement, "link_B2"
    )
    model.appendBodyToJoint(joint3_id, inertia_link_B, placement_center_link_B)
    geom_obj3 = pin.GeometryObject(
        "link_B2", joint3_id, shape_link_B, placement_shape_B
    )
    geom_obj3.meshColor = RED_COLOR
    collision_model.addGeometryObject(geom_obj3)

    q0 = pin.neutral(model)

    data = model.createData()
    pin.forwardKinematics(model, data, q0)

    # Set the contact constraints
    constraint1_joint1_placement = pin.SE3.Identity()
    constraint1_joint1_placement.translation = pin.XAxis * length_link_B

    constraint1_joint2_placement = pin.SE3.Identity()
    constraint1_joint2_placement.translation = -pin.XAxis * length_link_A / 2.0
    constraint_model = pin.RigidConstraintModel(
        pin.ContactType.CONTACT_3D,
        model,
        joint3_id,
        constraint1_joint1_placement,
        # model.jointPlacements[joint3_id],
        base_joint_id,
        constraint1_joint2_placement,
    )
    # model.jointPlacements[base_joint_id])
    constraint_model.corrector.Kp[:] = 10.0
    constraint_model.corrector.Kd[:] = 2.0 * np.sqrt(constraint_model.corrector.Kp)
    constraint_data = constraint_model.createData()
    constraint_dim = constraint_model.size()

    # First, do an inverse kinematics
    rho = 1e-10
    mu = 1e-4

    q = q0.copy()

    y = np.ones((constraint_dim))
    data.M = np.eye(model.nv) * rho
    kkt_constraint = pin.ContactCholeskyDecomposition(model, [constraint_model])
    eps = 1e-10
    N = 100
    for k in range(N):
        pin.computeJointJacobians(model, data, q)
        kkt_constraint.compute(model, data, [constraint_model], [constraint_data], mu)
        constraint_value = constraint_data.c1Mc2.translation

        J = pin.getFrameJacobian(
            model,
            data,
            constraint_model.joint1_id,
            constraint_model.joint1_placement,
            constraint_model.reference_frame,
        )[:3, :]
        primal_feas = np.linalg.norm(constraint_value, np.inf)
        dual_feas = np.linalg.norm(J.T.dot(constraint_value + y), np.inf)
        if primal_feas < eps and dual_feas < eps:
            print("Convergence achieved")
            break
        print("constraint_value: {:.3e}".format(np.linalg.norm(constraint_value)))
        rhs = np.concatenate([-constraint_value - y * mu, np.zeros(model.nv)])

        dz = kkt_constraint.solve(rhs)
        dy = dz[:constraint_dim]
        dq = dz[constraint_dim:]

        alpha = 1.0
        q = pin.integrate(model, q, -alpha * dq)
        y -= alpha * (-dy + y)

    q_sol = (q[:] + np.pi) % np.pi - np.pi
    model.q_init = q_sol
    return model, constraint_model


def test_inv_dyn():
    model, rcm = createFourBarLinkages()
    data = model.createData()
    rcd = rcm.createData()
    q0 = model.q_init.copy()
    v0 = np.zeros(model.nv)
    nu = model.nv
    B = np.eye(nu)
    tau0, _fc = aligator.underactuatedConstrainedInverseDynamics(
        model, data, q0, v0, B, [rcm], [rcd]
    )

    # test the resulting acceleration is zero
    pin.initConstraintDynamics(model, data, [rcm])
    prox = pin.ProximalSettings(1e-12, 1e-10, 3)
    a0 = pin.constraintDynamics(model, data, q0, v0, tau0, [rcm], [rcd], prox)
    assert np.allclose(a0, 0.0)


def test_constrained_dynamics():
    model, constraint_model = createFourBarLinkages()

    # check derivatives
    space = manifolds.MultibodyPhaseSpace(model)
    nu = model.nv
    B = np.eye(nu)
    # only control one DOF
    B[1:, 1:] = 0.0
    print(f"B matrix :\n{B}")
    prox = pin.ProximalSettings(1e-12, 1e-10, 3)

    ode = dynamics.MultibodyConstraintFwdDynamics(space, B, [constraint_model], prox)
    data = ode.createData()
    assert isinstance(data, dynamics.MultibodyConstraintFwdData)

    x0 = space.neutral()
    x0[: model.nq] = model.q_init
    u0 = np.random.randn(nu)

    ode.forward(x0, u0, data)
    ode.dForward(x0, u0, data)

    Jx_ = data.Jx
    Ju_ = data.Ju

    Jx, Ju = finite_diff(ode, space, x0, u0)
    err_Jx = np.max(Jx_ - Jx)
    err_Ju = np.max(Ju_ - Ju)

    assert err_Jx < TOL and err_Ju < TOL

    # Perform forward simulation of free swinging
    dt = 5e-3
    discrete_dynamics = dynamics.IntegratorSemiImplEuler(ode, dt)
    data = discrete_dynamics.createData()

    # target is following config
    # 0----0
    # |    |
    # 0----0
    x_target = np.array([-np.pi / 2] * 3 + [0] * 3)

    nsteps = 300
    stages = []

    us_init = [np.zeros(nu)] * nsteps
    xs_init = [x0] * (nsteps + 1)

    w_x = np.ones(x0.size)
    w_x[: model.nq] = 10
    w_u = np.eye(nu) * 1e-3

    for i in range(nsteps):
        rcost = aligator.CostStack(space, nu)
        xreg_cost = aligator.QuadraticStateCost(space, nu, x_target, np.diag(w_x) * dt)
        rcost.addCost(xreg_cost)

        ucost = aligator.QuadraticControlCost(space, nu, w_u * dt)
        rcost.addCost(ucost)

        stage = aligator.StageModel(rcost, discrete_dynamics)

        stages.append(stage)

    term_cost = aligator.QuadraticStateCost(space, nu, x_target, np.diag(w_x))

    problem = aligator.TrajOptProblem(x0, stages, term_cost=term_cost)

    tol = 1e-5
    mu_init = 0.01
    verbose = aligator.VerboseLevel.VERBOSE
    solver = aligator.SolverProxDDP(tol, mu_init, verbose=verbose, max_iters=200)
    solver.setup(problem)
    conv = solver.run(problem, xs_init, us_init)

    results = solver.results
    print(results)
    assert conv


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
