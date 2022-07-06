"""
@Time    :   2022/06/29 15:58:26
@Author  :   quentinll
@License :   (C)Copyright 2021-2022, INRIA
"""

import pinocchio as pin
import numpy as np
import proxddp
import proxnlp
import hppfcl as fcl
import tap
import matplotlib.pyplot as plt


class Args(tap.Tap):
    use_term_cstr: bool = True


args = Args().parse_args()


def create_cartpole(N):
    model = pin.Model()
    geom_model = pin.GeometryModel()

    parent_id = 0

    cart_radius = 0.1
    cart_length = 5 * cart_radius
    cart_mass = 1.0
    joint_name = "joint_cart"

    geometry_placement = pin.SE3.Identity()
    geometry_placement.rotation = pin.Quaternion(
        np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
    ).toRotationMatrix()

    joint_id = model.addJoint(
        parent_id, pin.JointModelPY(), pin.SE3.Identity(), joint_name
    )

    body_inertia = pin.Inertia.FromCylinder(cart_mass, cart_radius, cart_length)
    body_placement = geometry_placement
    model.appendBodyToJoint(
        joint_id, body_inertia, body_placement
    )  # We need to rotate the inertia as it is expressed in the LOCAL frame of the geometry

    shape_cart = fcl.Cylinder(cart_radius, cart_length)

    geom_cart = pin.GeometryObject(
        "shape_cart", joint_id, shape_cart, geometry_placement
    )
    geom_cart.meshColor = np.array([1.0, 0.1, 0.1, 1.0])
    geom_model.addGeometryObject(geom_cart)

    parent_id = joint_id
    joint_placement = pin.SE3.Identity()
    body_mass = 0.1
    body_radius = 0.1
    for k in range(N):
        joint_name = "joint_" + str(k + 1)
        joint_id = model.addJoint(
            parent_id, pin.JointModelRX(), joint_placement, joint_name
        )

        body_inertia = pin.Inertia.FromSphere(body_mass, body_radius)
        body_placement = joint_placement.copy()
        body_placement.translation[2] = 1.0
        model.appendBodyToJoint(joint_id, body_inertia, body_placement)

        geom1_name = "ball_" + str(k + 1)
        shape1 = fcl.Sphere(body_radius)
        geom1_obj = pin.GeometryObject(geom1_name, joint_id, shape1, body_placement)
        geom1_obj.meshColor = np.ones((4))
        geom_model.addGeometryObject(geom1_obj)

        geom2_name = "bar_" + str(k + 1)
        shape2 = fcl.Cylinder(body_radius / 4.0, body_placement.translation[2])
        shape2_placement = body_placement.copy()
        shape2_placement.translation[2] /= 2.0

        geom2_obj = pin.GeometryObject(geom2_name, joint_id, shape2, shape2_placement)
        geom2_obj.meshColor = np.array([0.0, 0.0, 0.0, 1.0])
        geom_model.addGeometryObject(geom2_obj)

        parent_id = joint_id
        joint_placement = body_placement.copy()
    end_frame = pin.Frame(
        "end_effector_frame",
        model.getJointId("joint_" + str(N)),
        0,
        body_placement,
        pin.FrameType(3),
    )
    model.addFrame(end_frame)
    geom_model.collision_pairs = []
    model.qinit = np.zeros(model.nq)
    model.qinit[1] = 0.0 * np.pi
    model.qref = pin.neutral(model)
    data = model.createData()
    geom_data = geom_model.createData()
    ddl = np.array([0])
    return model, geom_model, data, geom_data, ddl


model, geom_model, data, geom_data, ddl = create_cartpole(1)
time_step = 0.01
nu = 1
act_mat = np.zeros((2, nu))
act_mat[0, 0] = 1.0
space = proxnlp.manifolds.MultibodyPhaseSpace(model)
nx = space.nx
ndx = space.ndx
cont_dyn = proxddp.dynamics.MultibodyFreeFwdDynamics(space, act_mat)
disc_dyn = proxddp.dynamics.IntegratorSemiImplEuler(cont_dyn, time_step)

nq = model.nq
nv = model.nv
x0 = space.neutral()
x0[1] = np.pi

target_pos = np.array([0.0, 0.0, 1.0])
frame_id = model.getFrameId("end_effector_frame")

# running cost regularizes the control input
rcost = proxddp.CostStack(ndx, nu)
wu = np.ones(nu) * 1e-2
rcost.addCost(
    proxddp.QuadraticResidualCost(
        proxddp.ControlErrorResidual(ndx, nu, np.zeros(nu)), np.diag(wu)
    )
)
term_cost = proxddp.CostStack(ndx, nu)
stage = proxddp.StageModel(space, nu, rcost, disc_dyn)

# box constraint on control
u_min = -25.0 * np.ones(nu)
u_max = +25.0 * np.ones(nu)
ctrl_box = proxddp.ControlBoxFunction(ndx, u_min, u_max)
stage.addConstraint(
    proxddp.StageConstraint(ctrl_box, proxnlp.constraints.NegativeOrthant())
)


class FramePosErrorResidual(proxddp.StageFunction):
    """residual error on a frame position.
    temporarily replace the c++ implementation"""

    def __init__(self, model, nx, nu, target_pos, frame_id, terminal=True) -> None:
        super().__init__(nx, nu, 3)
        self.nx = nx
        self.target = target_pos
        self.rmodel = model
        self.rdata = model.createData()
        self.fid = frame_id
        self.terminal = terminal

    def evaluate(self, x, u, y, data):
        if self.terminal:
            x = y
        q = x[: self.rmodel.nq]
        pin.forwardKinematics(self.rmodel, self.rdata, q)
        pin.updateFramePlacements(self.rmodel, self.rdata)

        cur_pose = self.rdata.oMf[self.fid]
        cur_position = cur_pose.translation
        data.value[:] = cur_position - self.target

    def _get_errvec_and_jac(self, q):
        """Get the error vector and frame placement Jacobian."""
        pin.framesForwardKinematics(self.rmodel, self.rdata, q)
        pin.computeJointJacobians(self.rmodel, self.rdata)

        cur_pose = self.rdata.oMf[self.fid]
        cur_position = cur_pose.translation
        err = cur_position - self.target

        Jf = pin.getFrameJacobian(
            self.rmodel, self.rdata, self.fid, pin.LOCAL_WORLD_ALIGNED
        )
        Jf = Jf[:3]
        return err, Jf

    def computeJacobians(self, x, u, y, data):
        nq = self.rmodel.nq
        nv = self.rmodel.nv
        if self.terminal:
            x = y
        q = x[:nq]
        err, Jf = self._get_errvec_and_jac(q)
        if self.terminal:
            data.Jy[:, :nv] = Jf
        else:
            data.Jx[:, :nv] = Jf


nsteps = 600
Tf = nsteps * time_step
problem = proxddp.TrajOptProblem(x0, nu, space, term_cost)
for i in range(nsteps):
    if i == nsteps - 1 and args.use_term_cstr:
        xtar = space.neutral()
        # term_fun = FramePosErrorResidual(model, nx, nu, target_pos, frame_id, terminal=True)
        # stage.addConstraint(
        #     proxddp.StageConstraint(
        #         term_fun, proxnlp.constraints.EqualityConstraintSet()
        #     )
        # )
        stage.addConstraint(
            proxddp.StateErrorResidual(space, nu, xtar),
            proxnlp.constraints.EqualityConstraintSet(),
        )
    problem.addStage(stage)

mu_init = 1e-2
verbose = proxddp.VerboseLevel.VERBOSE
TOL = 1e-3
MAX_ITER = 500
solver = proxddp.ProxDDP(TOL, mu_init, max_iters=MAX_ITER, verbose=verbose)

u0 = np.zeros(nu)
us_i = [u0] * nsteps
xs_i = proxddp.rollout(disc_dyn, x0, us_i)
prob_data = proxddp.TrajOptData(problem)
problem.evaluate(xs_i, us_i, prob_data)

solver.setup(problem)
solver.run(problem, xs_i, us_i)
res = solver.getResults()

plt.figure(figsize=(9.6, 4.8))
plt.subplot(121)
lstyle = {"lw": 0.9}
trange = np.linspace(0, Tf, nsteps + 1)
plt.plot(trange, res.xs, ls="-", **lstyle)
plt.title("State $x(t)$")
if args.use_term_cstr:
    plt.hlines(
        xtar,
        *trange[[0, -1]],
        ls="-",
        lw=1.3,
        colors="k",
        alpha=0.8,
        label=r"$x_\mathrm{tar}$"
    )
plt.legend()
plt.xlabel("Time $i$")

plt.subplot(122)
plt.plot(trange[:-1], res.us, **lstyle)
plt.hlines(
    np.concatenate([u_min, u_max]),
    *trange[[0, -1]],
    ls="-",
    colors="k",
    lw=2.5,
    alpha=0.4,
    label=r"$\bar{u}$"
)
plt.title("Controls $u(t)$")

plt.legend()

plt.tight_layout()
plt.show()
