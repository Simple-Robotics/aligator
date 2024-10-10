"""
Simple quadrotor dynamics example.

Inspired by: https://github.com/loco-3d/crocoddyl/blob/master/examples/quadrotor.py
"""

import pinocchio as pin
import hppfcl as fcl
import example_robot_data as erd

import numpy as np
import matplotlib.pyplot as plt

import aligator

from aligator import manifolds
from proxsuite_nlp import constraints

from utils import ArgsBase, manage_lights

robot = erd.load("hector")
rmodel = robot.model
rdata = robot.data
nq = rmodel.nq
nv = rmodel.nv
ROT_NULL = np.eye(3)


class Args(ArgsBase):
    bounds: bool = False
    """Use control bounds"""
    obstacles: bool = False  # Obstacles in the environment
    random: bool = False
    term_cstr: bool = False
    fddp: bool = False


def create_halfspace_z(ndx, nu, offset: float = 0.0, neg: bool = False):
    r"""
    Constraint :math:`z \geq offset`.
    """
    root_frame_id = 1
    p_ref = np.zeros(3)
    frame_fun = aligator.FrameTranslationResidual(ndx, nu, rmodel, p_ref, root_frame_id)
    A = np.array([[0.0, 0.0, 1.0]])
    b = np.array([-offset])
    sign = -1.0 if neg else 1.0
    frame_fun_z = aligator.LinearFunctionComposition(frame_fun, sign * A, sign * b)
    return frame_fun_z


class Column(aligator.StageFunction):
    def __init__(
        self,
        rmodel: pin.Model,
        ndx,
        nu,
        center,
        radius,
        margin: float = 0.0,
    ) -> None:
        super().__init__(ndx, nu, 1)
        self.rmodel = rmodel.copy()
        self.rdata = self.rmodel.createData()
        self.ndx = ndx
        self.center = center.copy()
        self.radius = radius
        self.margin = margin

    def __getinitargs__(self):
        return (self.rmodel, self.ndx, self.nu, self.center, self.radius, self.margin)

    def evaluate(self, x, u, data):  # distance function
        q = x[:nq]
        pin.forwardKinematics(self.rmodel, self.rdata, q)
        M: pin.SE3 = pin.updateFramePlacement(self.rmodel, self.rdata, 1)
        err = M.translation[:2] - self.center
        res = np.dot(err, err) - (self.radius + self.margin) ** 2
        data.value[:] = -res

    def computeJacobians(self, x, u, data):
        q = x[:nq]
        J = pin.computeFrameJacobian(
            self.rmodel, self.rdata, q, 1, pin.LOCAL_WORLD_ALIGNED
        )
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
    from utils import ASSET_DIR

    print(args)

    if args.obstacles:  # we add the obstacles to the geometric model
        cyl_radius = 0.22
        cylinder = fcl.Cylinder(cyl_radius, 10.0)
        center_column1 = np.array([-0.45, 1.2, 0.0])
        center_column2 = np.array([0.4, 2.4, 0.0])

        geom_cyl1 = pin.GeometryObject(
            "column1", 0, cylinder, pin.SE3(ROT_NULL, center_column1)
        )
        geom_cyl2 = pin.GeometryObject(
            "column2", 0, cylinder, pin.SE3(ROT_NULL, center_column2)
        )
        cyl_color1 = np.array([1.0, 0.2, 1.0, 0.4])
        cyl_color2 = np.array([0.2, 1.0, 1.0, 0.4])
        geom_cyl1.meshColor = cyl_color1
        geom_cyl2.meshColor = cyl_color2
        robot.collision_model.addGeometryObject(geom_cyl1)
        robot.visual_model.addGeometryObject(geom_cyl1)
        robot.collision_model.addGeometryObject(geom_cyl2)
        robot.visual_model.addGeometryObject(geom_cyl2)

    if args.display:
        # 1st arg is the plane normal
        # 2nd arg is offset from origin
        plane = fcl.Plane(np.array([0.0, 0.0, 1.0]), 0.0)
        plane_obj = pin.GeometryObject("plane", 0, plane, pin.SE3.Identity())
        plane_obj.meshColor[:] = [1.0, 1.0, 0.95, 1.0]
        plane_obj.meshScale[:] = 2.0
        robot.visual_model.addGeometryObject(plane_obj)
        robot.collision_model.addGeometryObject(plane_obj)

    def add_objective_vis_models(x_tar1, x_tar2, x_tar3):
        """Add visual guides for the objectives."""
        objective_color = np.array([5, 104, 143, 200]) / 255.0
        if args.obstacles:
            sp1_obj = pin.GeometryObject(
                "obj1", 0, fcl.Sphere(0.05), pin.SE3(ROT_NULL, x_tar3[:3])
            )
            sp1_obj.meshColor[:] = objective_color
            robot.visual_model.addGeometryObject(sp1_obj)
        else:
            sp1_obj = pin.GeometryObject(
                "obj1", 0, fcl.Sphere(0.05), pin.SE3(ROT_NULL, x_tar1[:3])
            )
            sp2_obj = pin.GeometryObject(
                "obj2", 0, fcl.Sphere(0.05), pin.SE3(ROT_NULL, x_tar2[:3])
            )
            sp1_obj.meshColor[:] = objective_color
            sp2_obj.meshColor[:] = objective_color
            robot.visual_model.addGeometryObject(sp1_obj)
            robot.visual_model.addGeometryObject(sp2_obj)

    robot.collision_model.geometryObjects[0].geometry.computeLocalAABB()
    quad_radius = robot.collision_model.geometryObjects[0].geometry.aabb_radius

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

    space = manifolds.MultibodyPhaseSpace(rmodel)
    ode_dynamics = aligator.dynamics.MultibodyFreeFwdDynamics(space, QUAD_ACT_MATRIX)

    dt = 0.01
    Tf = 1.8
    nsteps = int(Tf / dt)
    print("nsteps: {:d}".format(nsteps))

    if args.integrator == "euler":
        dynmodel = aligator.dynamics.IntegratorEuler(ode_dynamics, dt)
    elif args.integrator == "semieuler":
        dynmodel = aligator.dynamics.IntegratorSemiImplEuler(ode_dynamics, dt)
    elif args.integrator == "rk2":
        dynmodel = aligator.dynamics.IntegratorRK2(ode_dynamics, dt)
    elif args.integrator == "midpoint":
        dynmodel = aligator.dynamics.IntegratorMidpoint(ode_dynamics, dt)
    else:
        raise ValueError()

    x0 = np.concatenate([robot.q0, np.zeros(nv)])
    x0[2] = 0.18
    if args.random and args.obstacles:
        x0[:3] = sample_feasible_translation(
            [center_column1, center_column2], cyl_radius, quad_radius
        )

    tau = pin.rnea(rmodel, rdata, robot.q0, np.zeros(nv), np.zeros(nv))
    u0, _, _, _ = np.linalg.lstsq(QUAD_ACT_MATRIX, tau, rcond=-1)

    us_init = [u0] * nsteps
    xs_init = aligator.rollout(dynmodel, x0, us_init)

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

            def _schedule(i):
                return weights, x_tar3

        else:
            # waypoint task
            weights1 = np.zeros(space.ndx)
            weights1[:3] = 4.0
            weights1[3:6] = 1e-2
            weights1[nv:] = 1e-3
            weights2 = weights1.copy()
            weights2[:3] = 1.0

            def _schedule(i):
                x_tar = x_tar1
                weights = weights1
                if i == idx_switch:
                    weights[:] /= dt
                if i > idx_switch:
                    x_tar = x_tar2
                    weights = weights2
                return weights, x_tar

        return _schedule

    task_schedule = get_task_schedule()

    def setup() -> aligator.TrajOptProblem:
        w_u = np.eye(nu) * 1e-1

        wterm, x_tar = task_schedule(nsteps)
        if not args.term_cstr:
            wterm *= 12.0
        term_cost = aligator.QuadraticStateCost(space, nu, x_tar, np.diag(wterm))
        prob = aligator.TrajOptProblem(x0, nu, space, term_cost=term_cost)

        floor = create_halfspace_z(space.ndx, nu, 0.0, True)
        if args.bounds:
            u_identity_fn = aligator.ControlErrorResidual(space.ndx, np.zeros(nu))
            box_set = constraints.BoxConstraint(u_min, u_max)
            ctrl_cstr = (u_identity_fn, box_set)

        for i in range(nsteps):
            rcost = aligator.CostStack(space, nu)

            weights, x_tar = task_schedule(i)

            xreg_cost = aligator.QuadraticStateCost(
                space, nu, x_tar, np.diag(weights) * dt
            )
            rcost.addCost(xreg_cost)
            ureg_cost = aligator.QuadraticControlCost(space, u0, w_u * dt)
            rcost.addCost(ureg_cost)

            stage = aligator.StageModel(rcost, dynmodel)
            if args.bounds:
                stage.addConstraint(*ctrl_cstr)
            if args.obstacles:  # add obstacles' constraints
                column1 = Column(
                    rmodel, space.ndx, nu, center_column1[:2], cyl_radius, quad_radius
                )
                stage.addConstraint(floor, constraints.NegativeOrthant())
                stage.addConstraint(column1, constraints.NegativeOrthant())
                column2 = Column(
                    rmodel, space.ndx, nu, center_column2[:2], cyl_radius, quad_radius
                )
                stage.addConstraint(column2, constraints.NegativeOrthant())
            prob.addStage(stage)
        if args.term_cstr:
            prob.addTerminalConstraint(
                aligator.StateErrorResidual(space, nu, x_tar),
                constraints.EqualityConstraintSet(),
            )
        return prob

    _, x_term = task_schedule(nsteps)
    problem = setup()

    if args.display:
        vizer = pin.visualize.MeshcatVisualizer(
            rmodel, robot.collision_model, robot.visual_model, data=rdata
        )
        vizer.initViewer(loadModel=True, zmq_url=args.zmq_url)
        manage_lights(vizer)
        vizer.display(x0[:nq])
    else:
        vizer = None

    tol = 1e-4
    mu_init = 1.0
    verbose = aligator.VerboseLevel.VERBOSE
    history_cb = aligator.HistoryCallback()
    solver = aligator.SolverProxDDP(tol, mu_init, verbose=verbose)
    if args.fddp:
        solver = aligator.SolverFDDP(tol, verbose=verbose)
    solver.max_iters = 400
    solver.registerCallback("his", history_cb)
    solver.bcl_params.dyn_al_scale = 1e-6
    solver.setup(problem)
    solver.run(problem, xs_init, us_init)

    results = solver.results
    print(results)

    xs_opt = results.xs.tolist()
    us_opt = results.us.tolist()

    def plot_costate_value() -> plt.Figure:
        costate_stack = np.stack(results.lams).T
        if solver.force_initial_condition:
            costate_stack[:, 0] = np.nan
        plt.figure()
        mmin = np.min(costate_stack)
        mmax = np.max(costate_stack)
        plt.imshow(costate_stack, vmin=mmin, vmax=mmax, aspect="auto")
        plt.vlines(idx_switch, *plt.ylim(), colors="r", label="switch")

        plt.legend()
        plt.xlabel("Time $t$")
        plt.ylabel("Dimension")
        plt.title("Multipliers")
        plt.colorbar()
        plt.tight_layout()
        return plt.gcf()

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
        n_iter = np.arange(len(history_cb.prim_infeas.tolist()))
        ax2.semilogy(
            n_iter[1:], history_cb.prim_infeas.tolist()[1:], label="Primal err."
        )
        ax2.semilogy(n_iter, history_cb.dual_infeas.tolist(), label="Dual err.")
        ax2.set_xlabel("Iterations")
        ax2.legend()

        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(ASSET_DIR / "{}.{}".format(TAG, ext))
        plt.show()

    if args.display:
        cam_dist = 2.0
        directions_ = [np.array([1.0, 1.0, 0.5])]
        directions_.append(np.array([1.0, -1.0, 0.8]))
        directions_.append(np.array([0.1, 0.1, 1.0]))
        directions_.append(np.array([0.0, -1.0, 0.8]))
        for d in directions_:
            d /= np.linalg.norm(d)

        vid_uri = ASSET_DIR / "{}.mp4".format(TAG)
        qs_opt = [x[:nq] for x in xs_opt]
        base_link_id = rmodel.getFrameId("base_link")

        def get_callback(i: int):
            pos_ema = np.zeros(3) + directions_[i] * cam_dist
            blend = 0.9

            def _callback(t):
                rp = xs_opt[t][:3]
                pos_new = rp + directions_[i] * cam_dist
                nonlocal pos_ema
                pos_ema = blend * pos_new + (1 - blend) * pos_ema
                vizer.setCameraPosition(pos_ema)
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
