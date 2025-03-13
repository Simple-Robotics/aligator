#include "aligator/core/traj-opt-problem.hpp"
#include "aligator/core/traj-opt-data.hpp"
#include "aligator/core/cost-abstract.hpp"
#include "aligator/core/stage-data.hpp"
#include "aligator/utils/mpc-util.hpp"
#include "aligator/solvers/proxddp/solver-proxddp.hpp"
#include "aligator/modelling/dynamics/multibody-constraint-fwd.hpp"
#include "aligator/modelling/dynamics/integrator-semi-euler.hpp"
#include "aligator/modelling/dynamics/integrator-rk2.hpp"
#include "aligator/modelling/multibody/frame-velocity.hpp"
#include "aligator/modelling/multibody/frame-placement.hpp"
#include "aligator/modelling/costs/sum-of-costs.hpp"
#include "aligator/modelling/costs/quad-state-cost.hpp"
#include "aligator/modelling/costs/constant-cost.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <boost/test/unit_test.hpp>

using namespace aligator;
using context::SolverProxDDP;

namespace {
pinocchio::Model SimpleModel() {
  pinocchio::Model model;

  auto const joint_id = model.addJoint(0, pinocchio::JointModelPZ(),
                                       pinocchio::SE3::Identity(), "joint");
  auto const frame_id = model.addJointFrame(joint_id, -1);
  model.addBodyFrame("link", joint_id, pinocchio::SE3::Identity(),
                     static_cast<int>(frame_id));

  Eigen::Vector3d const CoM = Eigen::Vector3d::Zero();
  double constexpr mass = 0.001;
  Eigen::Matrix3d const I = 0.001 * Eigen::Matrix3d::Identity();
  model.appendBodyToJoint(joint_id, pinocchio::Inertia(mass, CoM, I),
                          pinocchio::SE3::Identity());

  return model;
}

using StageSpace = proxsuite::nlp::MultibodyPhaseSpace<double>;

auto height(double t) -> std::tuple<pinocchio::SE3, bool> {
  double const t0 = std::floor(t);
  bool const support = static_cast<int>(t0) % 2 == 0;

  pinocchio::SE3 placement = pinocchio::SE3::Identity();

  if (!support) {
    double constexpr h = 0.02;
    double const dt = t - t0;
    placement.translation()[2] = h * 4. * dt * (1. - dt);
  }

  return {placement, support};
}

auto makePositionResidual(pinocchio::Model const &model,
                          pinocchio::SE3 const &placement) {
  return FramePlacementResidualTpl<double>(StageSpace(model).ndx(), model.nv,
                                           model, placement,
                                           model.getFrameId("link"));
}

auto makeCost(double t, pinocchio::Model const &model) {
  int const nq = model.nq;
  int const nv = model.nv;
  int const nx = nq + nv;
  int const nu = nv;
  auto const stage_space = StageSpace(model);
  auto rcost = CostStackTpl<double>(stage_space, nu);

  Eigen::VectorXd const x0 = Eigen::VectorXd::Zero(nx);
  Eigen::VectorXd const u0 = Eigen::VectorXd::Zero(nu);

  Eigen::MatrixXd w_x = Eigen::MatrixXd::Zero(nx, nx);
  w_x.diagonal().tail(nv).array() = 0.1;
  Eigen::MatrixXd const w_u = 0.01 * Eigen::MatrixXd::Identity(nu, nu);

  rcost.addCost("quad_state",
                QuadraticStateCostTpl<double>(stage_space, nu, x0, w_x));
  rcost.addCost("quad_control",
                QuadraticControlCostTpl<double>(stage_space, u0, w_u));

  auto const [placement, support] = height(t);

  if (!support) {
    Eigen::MatrixXd const w_height = 5000. * Eigen::MatrixXd::Identity(6, 6);
    auto const frame_fn = makePositionResidual(model, placement);
    rcost.addCost("frame_fn", QuadraticResidualCostTpl<double>(
                                  stage_space, frame_fn, w_height));
  }

  return rcost;
}

auto makeConstraint(pinocchio::Model const &model) {
  auto const &frame_id = model.getFrameId("link");
  auto const &frame = model.frames[frame_id];
  auto constraint = pinocchio::RigidConstraintModel(
      pinocchio::ContactType::CONTACT_6D, model, frame.parentJoint,
      frame.placement, pinocchio::LOCAL_WORLD_ALIGNED);
  constraint.corrector.Kp << 0, 0, 100, 0, 0, 0;
  constraint.corrector.Kd << 50, 50, 50, 50, 50, 50;
  constraint.name = "contact";

  return constraint;
}

auto makeDynamicsModel(double t, double dt, pinocchio::Model const &model) {
  auto constraint_models = typename dynamics::MultibodyConstraintFwdDynamicsTpl<
      double>::RigidConstraintModelVector();
  auto const [placement, support] = height(t);

  if (support)
    constraint_models.emplace_back(makeConstraint(model)).joint2_placement =
        placement;

  auto const stage_space = StageSpace(model);
  auto const proximal_settings = pinocchio::ProximalSettings(1e-9, 1e-10, 10);
  Eigen::MatrixXd actuation_matrix =
      Eigen::MatrixXd::Zero(model.nv, model.nv).eval();
  actuation_matrix.bottomRows(model.nv).setIdentity();

  auto const ode = dynamics::MultibodyConstraintFwdDynamicsTpl<double>(
      stage_space, actuation_matrix, constraint_models, proximal_settings);
  return dynamics::IntegratorSemiImplEulerTpl<double>(ode, dt);
}

auto makeStage(double t, double dt, pinocchio::Model const &model) {
  auto const rcost = makeCost(t, model);
  auto const dynModel = makeDynamicsModel(t, dt, model);

  return StageModelTpl<double>(rcost, dynModel);
}

} // namespace

BOOST_AUTO_TEST_CASE(test_simple_mpc) {

  auto const model = SimpleModel();
  auto data = pinocchio::Data(model);

  int const nq = model.nq;
  int const nv = model.nv;
  int const nx = nq + nv;
  int const nu = model.nv;
  int constexpr nsteps = 10;
  double constexpr dt = 0.02;

  fmt::print("nq: {:d}, nv: {:d}, nx: {:d}, nu: {:d}\n", nq, nv, nx, nu);

  Eigen::VectorXd q = pinocchio::neutral(model);
  Eigen::VectorXd vq = Eigen::VectorXd::Zero(nv);
  Eigen::VectorXd x = Eigen::VectorXd::Zero(nx);
  x << q, vq;

  pinocchio::computeAllTerms(model, data, q, vq);
  pinocchio::updateFramePlacements(model, data);

  auto const term_cost = makeCost(0., model);
  auto problem = TrajOptProblemTpl<double>(x, nu, StageSpace(model), term_cost);

  for (auto i = 0; i < nsteps; ++i)
    problem.addStage(makeStage(0., dt, model));

  double constexpr TOL = 1e-7;
  unsigned int constexpr max_iters = 100u;
  double constexpr mu_init = 1e-8;
  VerboseLevel constexpr verbosity = QUIET;

  auto ddp = SolverProxDDPTpl<double>(TOL, mu_init, max_iters, verbosity);
  ddp.rollout_type_ = RolloutType::LINEAR;
  ddp.sa_strategy_ = StepAcceptanceStrategy::LINESEARCH_NONMONOTONE;
  ddp.filter_.beta_ = 1e-5;
  ddp.force_initial_condition_ = true;
  ddp.reg_min = 1e-6;
  ddp.linear_solver_choice = LQSolverChoice::SERIAL;

  ddp.setup(problem);

  bool converged = ddp.run(problem);
  BOOST_CHECK(converged);

  for (auto t = 0.; t < 5.; t += dt) {
    auto const t0 = std::chrono::steady_clock::now();
    auto const x = ddp.results_.xs[1];
    q = x.head(nq);
    vq = x.tail(nv);
    pinocchio::computeAllTerms(model, data, q, vq);
    pinocchio::updateFramePlacements(model, data);

    auto const stage = makeStage(t, dt, model);
    problem.replaceStageCircular(stage);
    ddp.cycleProblem(problem, stage.createData());
    problem.term_cost_ = makeCost(t + dt, model);
    ddp.workspace_.problem_data.term_cost_data =
        problem.term_cost_->createData();
    problem.setInitState(x);

    bool converged = ddp.run(problem);
    BOOST_CHECK(converged);
    auto const [expected, support] =
        height(std::max(0., t - (nsteps - 1) * dt));
    auto const &actual = data.oMf[model.getFrameId("link")];
    BOOST_CHECK((actual.inverse() * expected).isIdentity(2e-3));
    BOOST_CHECK_SMALL(actual.translation()[2] - expected.translation()[2],
                      2e-3);

    auto const tf = std::chrono::steady_clock::now();
    auto const time = std::chrono::duration<double, std::milli>(tf - t0);
    fmt::print("Elapsed time: {} ms\n", time.count());
  }
}
