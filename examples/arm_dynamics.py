import proxddp
import example_robot_data as erd
import numpy as np

import pinocchio as pin
import meshcat_utils as msu
from pinocchio.visualize import MeshcatVisualizer
from proxddp import dynamics, manifolds
import tap


class Args(tap.Tap):
    display: bool = False


args = Args().parse_args()
print(args)

robot = erd.load("ur5")
model = robot.model
space = manifolds.MultibodyPhaseSpace(model)

vizer = MeshcatVisualizer(model, robot.collision_model, robot.visual_model)
vizer.initViewer(open=args.display, loadModel=True)


nq = model.nq
nv = model.nv

B = np.eye(nv)
timestep = 0.02
contdynamics = dynamics.MultibodyFreeFwdDynamics(space, B)
dyn_model = dynamics.IntegratorSemiImplEuler(contdynamics, timestep)

Tf = 7.0
nsteps = int(Tf / timestep)

x0 = space.rand()
q0 = x0[:nq]
v0 = x0[nq:]
v0[:] = 0.0
v0[0] = 0.2
v0[2] = 1.0
v0[-1] = 0.01
vizer.display(q0)

target_acc = np.zeros(model.nv)
target_acc[1] = 0.01
us_ref = []
xs_ref = [x0]
ddata = dyn_model.createData()
for i in range(nsteps):
    x = xs_ref[i]
    u = pin.rnea(model, robot.data, x[:nq], x[nq:], target_acc)
    dyn_model.forward(x, u, ddata)
    xs_ref.append(ddata.xout.copy())
    us_ref.append(u)
    print(x)


def test_impl_rollout():
    # rollout using implicit type
    xs2 = proxddp.rollout_implicit(space, dyn_model, x0, us_ref).tolist()
    close = True
    for (a, b) in zip(xs_ref, xs2):
        close = close & np.allclose(a, b)
    assert close


if __name__ == "__main__":
    test_impl_rollout()

    if args.display:
        input()

        for _ in range(3):
            msu.play_trajectory(vizer, xs_ref, us_ref)
