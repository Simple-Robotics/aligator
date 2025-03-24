#include "croc-talos-arm.hpp"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/model.hpp>

#include <crocoddyl/core/fwd.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>
#include <crocoddyl/multibody/actuations/full.hpp>
#include <crocoddyl/multibody/actions/free-fwddyn.hpp>
#include <crocoddyl/core/integrator/euler.hpp>
#include <crocoddyl/core/costs/cost-sum.hpp>
#include <crocoddyl/core/costs/residual.hpp>
#include <crocoddyl/core/utils/callbacks.hpp>
#include <crocoddyl/multibody/residuals/frame-placement.hpp>
#include <crocoddyl/multibody/residuals/state.hpp>
#include <crocoddyl/core/residuals/control.hpp>
#include <crocoddyl/core/solvers/fddp.hpp>

void makeTalosArm(pin::Model &model) {
  const std::string talos_arm_path =
      EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_left_arm.urdf";
  pin::urdf::buildModel(talos_arm_path, model);
}

std::shared_ptr<croc::ShootingProblem>
defineCrocoddylProblem(std::size_t nsteps) {
  using croc::ActuationModelFull;
  using croc::CostModelResidual;
  using croc::CostModelSum;
  using croc::IntegratedActionModelEuler;
  using croc::ResidualModelControl;
  using croc::ResidualModelFramePlacement;
  using croc::ResidualModelState;
  using croc::StateMultibody;
  using Eigen::VectorXd;
  using DAM = croc::DifferentialActionModelFreeFwdDynamics;
  using ActionModel = croc::ActionModelAbstract;

  auto rmodel = std::make_shared<pin::Model>();
  makeTalosArm(*rmodel);
  auto state = std::make_shared<StateMultibody>(rmodel);

  auto runningCost = std::make_shared<CostModelSum>(state);
  auto terminalCost = std::make_shared<CostModelSum>(state);

  pin::JointIndex joint_id = rmodel->getFrameId("gripper_left_joint");
  pin::SE3 target_frame;
  target_frame.setIdentity();
  target_frame.translation() << 0., 0., 0.4;

  auto framePlacementResidual = std::make_shared<ResidualModelFramePlacement>(
      state, joint_id, target_frame);

  auto goalTrackingCost =
      std::make_shared<CostModelResidual>(state, framePlacementResidual);
  auto xregCost = std::make_shared<CostModelResidual>(
      state, std::make_shared<ResidualModelState>(state));
  auto uregCost = std::make_shared<CostModelResidual>(
      state, std::make_shared<ResidualModelControl>(state));

  runningCost->addCost("gripperPose", goalTrackingCost, 1.0);
  runningCost->addCost("xReg", xregCost, 1e-4);
  runningCost->addCost("uReg", uregCost, 1e-4);

  terminalCost->addCost("gripperPose", goalTrackingCost, 1.0);

  auto actuationModel = std::make_shared<ActuationModelFull>(state);

  const double dt = 1e-3;

  VectorXd armature(7);
  armature << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0;

  auto contDyn = std::make_shared<DAM>(state, actuationModel, runningCost);
  contDyn->set_armature(armature);
  auto runningModel = std::make_shared<IntegratedActionModelEuler>(contDyn, dt);

  auto termContDyn = std::make_shared<DAM>(state, actuationModel, terminalCost);
  termContDyn->set_armature(armature);
  auto terminalModel =
      std::make_shared<IntegratedActionModelEuler>(termContDyn, 0.0);

  VectorXd q0(rmodel->nq);
  q0 << 0.173046, 1.0, -0.52366, 0.0, 0.0, 0.1, -0.005;
  VectorXd x0(state->get_nx());
  x0 << q0, VectorXd::Zero(rmodel->nv);

  std::vector<std::shared_ptr<ActionModel>> running_models(nsteps,
                                                           runningModel);

  auto shooting_problem = std::make_shared<croc::ShootingProblem>(
      x0, running_models, terminalModel);
  return shooting_problem;
}

void getInitialGuesses(
    const std::shared_ptr<croc::ShootingProblem> &croc_problem,
    std::vector<Eigen::VectorXd> &xs_i, std::vector<Eigen::VectorXd> &us_i) {
  using Eigen::VectorXd;

  const std::size_t nsteps = croc_problem->get_T();
  const auto &x0 = croc_problem->get_x0();
  const long nu = (long)croc_problem->get_nu_max();
  VectorXd u0 = VectorXd::Zero(nu);

  xs_i.assign(nsteps + 1, x0);
  us_i.assign(nsteps, u0);
}
