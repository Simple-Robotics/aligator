"""
Simple quadrotor dynamics example.

Inspired by: https://github.com/loco-3d/crocoddyl/blob/master/examples/quadrotor.py
"""
import pinocchio as pin
import hppfcl as fcl
import example_robot_data as erd

import numpy as np
import matplotlib.pyplot as plt

import os
import proxddp

from proxddp import manifolds
from proxnlp import constraints

from common import ArgsBase

robot = erd.load("hector")
rmodel = robot.model
rdata = robot.data
nq = rmodel.nq
nv = rmodel.nv
ROT_NULL = np.eye(3)


class Args(ArgsBase):
    integrator = "semieuler"
    bounds: bool = False
    """Use control bounds"""
    plot: bool = False  # Plot the trajectories
    display: bool = False
    obstacles: bool = False  # Obstacles in the environment
    random: bool = False
    term_cstr: bool = False
    fddp: bool = False

    def process_args(self):
        if self.record:
            self.display = True


def create_halfspace_z(ndx, nu, offset: float = 0.0, neg: bool = False):
    r"""
    Constraint :math:`z \geq offset`.
    """
    root_frame_id = 1
    p_ref = np.zeros(3)
    frame_fun = proxddp.FrameTranslationResidual(ndx, nu, rmodel, p_ref, root_frame_id)
    A = np.array([[0.0, 0.0, 1.0]])
    b = np.array([-offset])
    sign = -1.0 if neg else 1.0
    frame_fun_z = proxddp.LinearFunctionComposition(frame_fun, sign * A, sign * b)
    return frame_fun_z


class Column(proxddp.StageFunction):
    def __init__(self, ndx, nu, center, radius, margin: float = 0.0) -> None:
        super().__init__(ndx, nu, 1)
        self.ndx = ndx
        self.center = center.copy()
        self.radius = radius
        self.margin = margin

    def evaluate(self, x, u, y, data):  # distance function
        q = x[:nq]
        pin.forwardKinematics(rmodel, rdata, q)
        M: pin.SE3 = pin.updateFramePlacement(rmodel, rdata, 1)
        err = M.translation[:2] - self.center
        res = np.dot(err, err) - (self.radius + self.margin) ** 2
        data.value[:] = -res

    def computeJacobians(self, x, u, y, data):
        q = x[:nq]
        J = pin.computeFrameJacobian(rmodel, rdata, q, 1, pin.LOCAL_WORLD_ALIGNED)
        err = x[:2] - self.center
        data.Jx[:nv] = -2 * J[:2].T @ err


def is_feasible(point, centers, radius, margin):
    if len(centers) >= 1 and radius > 0:
        for i in range(len(centers)):
            dist = np.linalg.norm(point[:2] - centers[i][:2])
            if dist < radius + margin:
                return False
    return True


def sample_feasible_translation(centers, radius, margin):
    translation = np.random.uniform([-1.5, 0.0, 0.2], [2.0, 2.0, 1.0], 3)
    feas = is_feasible(translation, centers, radius, margin)
    while not feas:
        translation = np.random.uniform([-1.0, 0.0, 0.2], [2.0, 2.0, 1.0], 3)
        feas = is_feasible(translation, centers, radius, margin)
    return translation


def main(args: Args):
    import meshcat

    os.makedirs("assets", exist_ok=True)
    print(args)

    if args.obstacles:  # we add the obstacles to the geometric model
        cyl_radius = 0.22
        cylinder = fcl.Cylinder(cyl_radius, 10.0)
        center_column1 = np.array([-0.45, 0.8, 0.0])
        geom_cyl1 = pin.GeometryObject(
            "column1", 0, pin.SE3(ROT_NULL, center_column1), cylinder
        )
        center_column2 = np.array([0.3, 2.4, 0.0])
        geom_cyl2 = pin.GeometryObject(
            "column2", 0, pin.SE3(ROT_NULL, center_column2), cylinder
        )
        cyl_color = np.array([2.0, 0.2, 1.0, 0.4])
        geom_cyl1.meshColor = cyl_color
        geom_cyl2.meshColor = cyl_color
        robot.collision_model.addGeometryObject(geom_cyl1)
        robot.visual_model.addGeometryObject(geom_cyl1)
        robot.collision_model.addGeometryObject(geom_cyl2)
        robot.visual_model.addGeometryObject(geom_cyl2)

    if args.display:
        # 1st arg is the plane normal
        # 2nd arg is offset from origin
        plane = fcl.Plane(np.array([0.0, 0.0, 1.0]), 0.0)
        plane_obj = pin.GeometryObject("plane", 0, pin.SE3.Identity(), plane)
        plane_obj.meshColor[:] = [1.0, 1.0, 0.95, 1.0]
        plane_obj.meshScale[:] = 2.0
        robot.visual_model.addGeometryObject(plane_obj)
        robot.collision_model.addGeometryObject(plane_obj)

    def add_objective_vis_models(x_tar1, x_tar2, x_tar3):
        """Add visual guides for the objectives."""
        objective_color = np.array([5, 104, 143, 200]) / 255.0
        if args.obstacles:
            sp1_obj = pin.GeometryObject(
                "obj1", 0, pin.SE3(ROT_NULL, x_tar3[:3]), fcl.Sphere(0.05)
            )
            sp1_obj.meshColor[:] = objective_color
            robot.visual_model.addGeometryObject(sp1_obj)
        else:
            sp1_obj = pin.GeometryObject(
                "obj1", 0, pin.SE3(ROT_NULL, x_tar1[:3]), fcl.Sphere(0.05)
            )
            sp2_obj = pin.GeometryObject(
                "obj2", 0, pin.SE3(ROT_NULL, x_tar2[:3]), fcl.Sphere(0.05)
            )
            sp1_obj.meshColor[:] = objective_color
            sp2_obj.meshColor[:] = objective_color
            robot.visual_model.addGeometryObject(sp1_obj)
            robot.visual_model.addGeometryObject(sp2_obj)

    robot.collision_model.geometryObjects[0].geometry.computeLocalAABB()
    quad_radius = robot.collision_model.geometryObjects[0].geometry.aabb_radius

    space = manifolds.MultibodyPhaseSpace(rmodel)

    # The matrix below maps rotor controls to torques

    d_cog, cf, cm, u_lim, _ = 0.1525, 6.6e-5, 1e-6, 5.0, 0.1
    QUAD_ACT_MATRIX = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, d_cog, 0.0, -d_cog],
            [-d_cog, 0.0, d_cog, 0.0],
            [-cm / cf, cm / cf, -cm / cf, cm / cf],
        ]
    )
    nu = QUAD_ACT_MATRIX.shape[1]  # = no. of nrotors

    ode_dynamics = proxddp.dynamics.MultibodyFreeFwdDynamics(space, QUAD_ACT_MATRIX)

    dt = 0.033
    Tf = 1.5
    nsteps = int(Tf / dt)
    print("nsteps: {:d}".format(nsteps))

    if args.integrator == "euler":
        dynmodel = proxddp.dynamics.IntegratorEuler(ode_dynamics, dt)
    elif args.integrator == "semieuler":
        dynmodel = proxddp.dynamics.IntegratorSemiImplEuler(ode_dynamics, dt)
    elif args.integrator == "rk2":
        dynmodel = proxddp.dynamics.IntegratorRK2(ode_dynamics, dt)
    elif args.integrator == "midpoint":
        dynmodel = proxddp.dynamics.IntegratorMidpoint(ode_dynamics, dt)
    else:
        raise ValueError()

    x0 = np.concatenate([robot.q0, np.zeros(nv)])
    x0[2] = 0.18
    if args.random and args.obstacles:
        x0[:3] = sample_feasible_translation(
            [center_column1, center_column2], cyl_radius, quad_radius
        )

    tau = pin.rnea(rmodel, rdata, robot.q0, np.zeros(nv), np.zeros(nv))
    u0, _, _, _ = np.linalg.lstsq(QUAD_ACT_MATRIX, tau)

    us_init = [u0] * nsteps
    xs_init = [x0] * (nsteps + 1)

    x_tar1 = space.neutral()
    x_tar1[:3] = (0.9, 0.8, 1.0)
    x_tar2 = space.neutral()
    x_tar2[:3] = (1.4, -0.6, 1.0)
    x_tar3 = space.neutral()
    x_tar3[:3] = (-0.1, 3.2, 1.0)
    add_objective_vis_models(x_tar1, x_tar2, x_tar3)

    u_max = u_lim * np.ones(nu)
    u_min = np.zeros(nu)

    times = np.linspace(0, Tf, nsteps + 1)
    idx_switch = int(0.7 * nsteps)
    times_wp = [times[idx_switch], times[-1]]

    def get_task_schedule():
        if args.obstacles:
            weights = np.zeros(space.ndx)
            weights[:3] = 0.1
            weights[3:6] = 1e-2
            weights[nv:] = 1e-3

            def weight_target_selector(i):
                return weights, x_tar3

        else:
            weights1 = np.zeros(space.ndx)
            weights1[:3] = 4.0
            weights1[3:6] = 1e-2
            weights1[nv:] = 1e-3
            weights2 = weights1.copy()
            weights2[:3] = 1.0

            def weight_target_selector(i):
                x_tar = x_tar1
                weights = weights1
                if i == idx_switch:
                    weights[:] /= dt
                if i > idx_switch:
                    x_tar = x_tar2
                    weights = weights2
                return weights, x_tar

        return weight_target_selector

    task_schedule = get_task_schedule()

    def setup():

        w_u = np.eye(nu) * 1e-2

        ceiling = create_halfspace_z(space.ndx, nu, 2.0)
        floor = create_halfspace_z(space.ndx, nu, 0.0, True)
        stages = []
        if args.bounds:
            u_identity_fn = proxddp.ControlErrorResidual(space.ndx, np.zeros(nu))
            box_set = constraints.BoxConstraint(u_min, u_max)
            ctrl_cstr = proxddp.StageConstraint(u_identity_fn, box_set)

        for i in range(nsteps):

            rcost = proxddp.CostStack(space.ndx, nu)

            weights, x_tar = task_schedule(i)

            state_err = proxddp.StateErrorResidual(space, nu, x_tar)
            xreg_cost = proxddp.QuadraticResidualCost(state_err, np.diag(weights) * dt)

            rcost.addCost(xreg_cost)

            u_err = proxddp.ControlErrorResidual(space.ndx, nu)
            ucost = proxddp.QuadraticResidualCost(u_err, w_u * dt)
            rcost.addCost(ucost)

            stage = proxddp.StageModel(rcost, dynmodel)
            if args.bounds:
                stage.addConstraint(ctrl_cstr)
            if args.obstacles:  # add obstacles' constraints
                column1 = Column(
                    space.ndx, nu, center_column1[:2], cyl_radius, quad_radius
                )
                column2 = Column(
                    space.ndx, nu, center_column2[:2], cyl_radius, quad_radius
                )
                stage.addConstraint(ceiling, constraints.NegativeOrthant())
                stage.addConstraint(floor, constraints.NegativeOrthant())
                stage.addConstraint(column1, constraints.NegativeOrthant())
                stage.addConstraint(column2, constraints.NegativeOrthant())
            stages.append(stage)

        weights, x_tar = task_schedule(nsteps)
        if not args.term_cstr:
            weights *= 10.0
        term_cost = proxddp.QuadraticResidualCost(
            proxddp.StateErrorResidual(space, nu, x_tar), np.diag(weights)
        )
        prob = proxddp.TrajOptProblem(x0, stages, term_cost=term_cost)
        if args.term_cstr:
            term_cstr = proxddp.StageConstraint(
                proxddp.StateErrorResidual(space, nu, x_tar),
                constraints.EqualityConstraintSet(),
            )
            prob.addTerminalConstraint(term_cstr)
        return prob

    _, x_term = task_schedule(nsteps)
    problem = setup()

    viewer = meshcat.Visualizer()
    vizer = pin.visualize.MeshcatVisualizer(
        rmodel, robot.collision_model, robot.visual_model, data=rdata
    )
    vizer.initViewer(viewer, loadModel=True, open=args.display)
    vizer.displayCollisions(True)
    vizer.display(x0[:nq])

    tol = 1e-3
    mu_init = 1e-1
    rho_init = 0.0
    verbose = proxddp.VerboseLevel.VERBOSE
    history_cb = proxddp.HistoryCallback()
    solver = proxddp.SolverProxDDP(tol, mu_init, rho_init, verbose=verbose)
    if args.fddp:
        solver = proxddp.SolverFDDP(tol, verbose=verbose)
    solver.max_iters = 200
    solver.registerCallback("his", history_cb)
    solver.setup(problem)
    solver.run(problem, xs_init, us_init)

    results = solver.getResults()
    workspace = solver.getWorkspace()
    print(results)

    def test_check_numiters(results):
        if args.bounds:
            if args.obstacles:
                if args.term_cstr:
                    pass
                else:
                    assert results.num_iters <= 50
            else:
                if args.term_cstr:
                    assert results.num_iters <= 129
                else:
                    assert results.num_iters <= 33
        elif args.term_cstr:
            if args.obstacles:
                assert results.num_iters <= 39
            else:
                assert results.num_iters <= 20

    test_check_numiters(results)

    xs_opt = results.xs.tolist()
    us_opt = results.us.tolist()

    val_grad = [vp.Vx for vp in workspace.value_params]

    def plot_costate_value() -> plt.Figure:
        lams_stack = np.stack([la[: space.ndx] for la in results.lams]).T
        costate_stack = lams_stack[:, 1 : nsteps + 1]
        vx_stack = np.stack(val_grad).T[:, 1:]
        plt.figure()
        plt.subplot(131)
        mmin = min(np.min(costate_stack), np.min(vx_stack))
        mmax = max(np.max(costate_stack), np.max(vx_stack))
        plt.imshow(costate_stack, vmin=mmin, vmax=mmax, aspect="auto")
        plt.vlines(idx_switch, *plt.ylim(), colors="r", label="switch")
        plt.legend()

        plt.xlabel("Time $t$")
        plt.ylabel("Dimension")
        plt.title("Multipliers")
        plt.subplot(132)
        plt.imshow(vx_stack, vmin=mmin, vmax=mmax, aspect="auto")
        plt.colorbar()
        plt.xlabel("Time $t$")
        plt.ylabel("Dimension")
        plt.title("$\\nabla_xV$")

        plt.subplot(133)
        err = np.abs(costate_stack - vx_stack)
        plt.imshow(err, cmap="Reds", aspect="auto")
        plt.title("$\\lambda - V'_x$")
        plt.colorbar()
        plt.tight_layout()
        return plt.gcf()

    def test_results():
        assert results.num_iters == 91
        assert results.traj_cost <= 8.825e-01

    if args.obstacles:
        TAG = "quadrotor_obstacles"
    else:
        TAG = "quadrotor"

    root_pt_opt = np.stack(xs_opt)[:, :3]
    if args.plot:

        if len(results.lams) > 0:
            plot_costate_value()

        nplot = 3
        fig: plt.Figure = plt.figure(figsize=(9.6, 5.4))
        ax0: plt.Axes = fig.add_subplot(1, nplot, 1)
        ax0.plot(times[:-1], us_opt)
        ax0.hlines((u_min[0], u_max[0]), *times[[0, -1]], colors="k", alpha=0.3, lw=1.4)
        ax0.set_title("Controls")
        ax0.set_xlabel("Time")
        ax1: plt.Axes = fig.add_subplot(1, nplot, 2)
        ax1.plot(times, root_pt_opt)
        plt.legend(["$x$", "$y$", "$z$"])
        ax1.scatter([times_wp[-1]] * 3, x_term[:3], marker=".", c=["C0", "C1", "C2"])
        ax2: plt.Axes = fig.add_subplot(1, nplot, 3)
        n_iter = np.arange(len(history_cb.storage.prim_infeas.tolist()))
        ax2.semilogy(
            n_iter[1:], history_cb.storage.prim_infeas.tolist()[1:], label="Primal err."
        )
        ax2.semilogy(n_iter, history_cb.storage.dual_infeas.tolist(), label="Dual err.")
        ax2.set_xlabel("Iterations")
        ax2.legend()

        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig("assets/{}.{}".format(TAG, ext))
        plt.show()

    if args.display:
        cam_dist = 2.0
        directions_ = [np.array([1.0, 1.0, 0.5])]
        directions_.append(np.array([1.0, -1.0, 0.8]))
        directions_.append(np.array([0.1, 0.1, 1.0]))
        directions_.append(np.array([0.0, -1.0, 0.8]))
        for d in directions_:
            d /= np.linalg.norm(d)

        vid_uri = "assets/{}.mp4".format(TAG)
        qs_opt = [x[:nq] for x in xs_opt]
        base_link_id = rmodel.getFrameId("base_link")

        def get_callback(i: int):
            def _callback(t):
                n = len(root_pt_opt)
                n = min(t, n)
                rp = root_pt_opt[n]
                pos = rp + directions_[i] * cam_dist
                vizer.setCameraPosition(pos)
                vizer.setCameraTarget(rp)
                vel = xs_opt[t][nq:]
                pin.forwardKinematics(rmodel, vizer.data, qs_opt[t], vel)
                vizer.drawFrameVelocities(base_link_id)

            return _callback

        input("[enter to play]")
        if args.record:
            ctx = vizer.create_video_ctx(vid_uri, fps=30)
        else:
            import contextlib

            ctx = contextlib.nullcontext()
        with ctx:
            for i in range(4):
                vizer.play(qs_opt, dt, get_callback(i))


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
