/// @file
/// @brief Benchmark proxddp::SolverFDDP against Crocoddyl on a simple example

#include "proxddp/compat/crocoddyl/problem-wrap.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <example-robot-data/path.hpp>

#include <crocoddyl/multibody/states/multibody.hpp>
#include <crocoddyl/multibody/actuations/full.hpp>
#include <crocoddyl/multibody/actions/free-fwddyn.hpp>
#include <crocoddyl/core/integrator/euler.hpp>
#include <crocoddyl/core/costs/cost-sum.hpp>
#include <crocoddyl/core/costs/residual.hpp>
#include <crocoddyl/multibody/residuals/frame-placement.hpp>
#include <crocoddyl/multibody/residuals/state.hpp>
#include <crocoddyl/core/residuals/control.hpp>
#include <crocoddyl/core/solvers/fddp.hpp>

#include "proxddp/fddp/solver-fddp.hpp"

#include <benchmark/benchmark.h>

constexpr double TOL = 1e-4;
constexpr double TOL_proxddp = 2.2231e-4;
constexpr std::size_t maxiters = 100;

namespace pin = pinocchio;
namespace croc = crocoddyl;
using Eigen::MatrixXd;
using Eigen::VectorXd;

void makeTalosArm(pin::Model &model) {
  const std::string talos_arm_path =
      EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_left_arm.urdf";
  pin::urdf::buildModel(talos_arm_path, model);
}

/// This reimplements the Crocoddyl problem defined in
/// examples/croc_arm_manipulation.py.
boost::shared_ptr<croc::ShootingProblem> defineCrocoddylProblem() {
  using croc::ActuationModelFull;
  using croc::CostModelResidual;
  using croc::CostModelSum;
  using croc::IntegratedActionModelEuler;
  using croc::ResidualModelControl;
  using croc::ResidualModelFramePlacement;
  using croc::ResidualModelState;
  using croc::StateMultibody;
  using DAM = croc::DifferentialActionModelFreeFwdDynamics;
  using ActionModel = croc::ActionModelAbstract;

  auto rmodel = boost::make_shared<pin::Model>();
  makeTalosArm(*rmodel);
  auto state = boost::make_shared<StateMultibody>(rmodel);

  auto runningCost = boost::make_shared<CostModelSum>(state);
  auto terminalCost = boost::make_shared<CostModelSum>(state);

  pin::JointIndex joint_id = rmodel->getFrameId("gripper_left_joint");
  pin::SE3 target_frame(Eigen::Matrix3d::Identity(),
                        Eigen::Vector3d{0., 0., 0.4});

  auto framePlacementResidual = boost::make_shared<ResidualModelFramePlacement>(
      state, joint_id, target_frame);

  auto goalTrackingCost =
      boost::make_shared<CostModelResidual>(state, framePlacementResidual);
  auto xregCost = boost::make_shared<CostModelResidual>(
      state, boost::make_shared<ResidualModelState>(state));
  auto uregCost = boost::make_shared<CostModelResidual>(
      state, boost::make_shared<ResidualModelControl>(state));

  runningCost->addCost("gripperPose", goalTrackingCost, 1.0);
  runningCost->addCost("xReg", xregCost, 1e-4);
  runningCost->addCost("uReg", uregCost, 1e-4);

  terminalCost->addCost("gripperPose", goalTrackingCost, 1.0);

  auto actuationModel = boost::make_shared<ActuationModelFull>(state);

  const double dt = 1e-3;

  VectorXd armature(7);
  armature << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0;

  auto contDyn = boost::make_shared<DAM>(state, actuationModel, runningCost);
  contDyn->set_armature(armature);
  auto runningModel =
      boost::make_shared<IntegratedActionModelEuler>(contDyn, dt);

  auto termContDyn =
      boost::make_shared<DAM>(state, actuationModel, terminalCost);
  termContDyn->set_armature(armature);
  auto terminalModel =
      boost::make_shared<IntegratedActionModelEuler>(termContDyn, 0.0);

  const std::size_t nsteps = 50;

  VectorXd q0(rmodel->nq);
  q0 << 0.173046, 1.0, -0.52366, 0.0, 0.0, 0.1, -0.005;
  VectorXd x0(state->get_nx());
  x0 << q0, VectorXd::Zero(rmodel->nv);

  std::vector<boost::shared_ptr<ActionModel>> running_models(nsteps,
                                                             runningModel);

  auto shooting_problem = boost::make_shared<croc::ShootingProblem>(
      x0, running_models, terminalModel);
  return shooting_problem;
}

void getInitialGuesses(
    const boost::shared_ptr<croc::ShootingProblem> &croc_problem,
    std::vector<VectorXd> &xs_i, std::vector<VectorXd> &us_i) {

  const auto nsteps = croc_problem->get_T();
  const auto &x0 = croc_problem->get_x0();
  const auto nu = croc_problem->get_nu_max();
  VectorXd u0 = VectorXd::Zero(nu);

  xs_i.assign(nsteps + 1, x0);
  us_i.assign(nsteps, u0);
}

static void
BM_croc_fddp(benchmark::State &state,
             const boost::shared_ptr<croc::ShootingProblem> &croc_problem) {
  state.PauseTiming();
  std::vector<VectorXd> xs_i;
  std::vector<VectorXd> us_i;
  getInitialGuesses(croc_problem, xs_i, us_i);
  const std::size_t nsteps = croc_problem->get_T();

  croc::SolverFDDP solver(croc_problem);
  const double croc_tol = TOL * TOL * (double)nsteps;
  solver.set_th_stop(croc_tol);

  state.ResumeTiming();

  for (auto _ : state) {
    solver.solve(xs_i, us_i, maxiters);
  }
}

static void
BM_prox_fddp(benchmark::State &state,
             const boost::shared_ptr<croc::ShootingProblem> &croc_problem) {
  namespace compat = proxddp::compat::croc;

  state.PauseTiming();
  auto prob_wrap = compat::convertCrocoddylProblem(croc_problem);

  std::vector<VectorXd> xs_i;
  std::vector<VectorXd> us_i;
  getInitialGuesses(croc_problem, xs_i, us_i);

  auto verbose = proxddp::VerboseLevel::QUIET;
  proxddp::SolverFDDP<double> solver(TOL_proxddp, verbose);
  solver.setup(prob_wrap);

  state.ResumeTiming();

  for (auto _ : state) {
    solver.run(prob_wrap, xs_i, us_i);
  }
}

int main(int argc, char **argv) {
  auto croc_problem = defineCrocoddylProblem();

  auto unit = benchmark::kMillisecond;
  benchmark::RegisterBenchmark("croc::FDDP", &BM_croc_fddp, croc_problem)
      ->Unit(unit);
  benchmark::RegisterBenchmark("proxddp::FDDP", &BM_prox_fddp, croc_problem)
      ->Unit(unit);
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
}
