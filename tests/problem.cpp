#include "aligator/core/traj-opt-problem.hpp"
#include "aligator/core/traj-opt-data.hpp"
#include "aligator/core/stage-data.hpp"
#include "aligator/solvers/proxddp/solver-proxddp.hpp"
#include "aligator/solvers/proxddp/results.hpp"
#include "aligator/solvers/proxddp/workspace.hpp"
#include "aligator/core/explicit-dynamics.hpp"
#include "aligator/core/cost-abstract.hpp"
#include "aligator/utils/rollout.hpp"
#include "aligator/modelling/constraints/equality-constraint.hpp"
#ifdef ALIGATOR_WITH_PINOCCHIO
#include "aligator/modelling/spaces/pinocchio-groups.hpp"
#else
#include "aligator/core/vector-space.hpp"
#endif
#include "aligator/third-party/polymorphic_cxx14.h"
// #include <boost/test/unit_test.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>

using namespace aligator;

/// @brief    Addition dynamics.
/// @details  It maps \f$(x,u)\f$ to \f$ x + u \f$.
struct MyModel : ExplicitDynamicsModelTpl<double> {
  using Manifold = ManifoldAbstractTpl<double>;
  using ManifoldPtr = xyz::polymorphic<Manifold>;
  using ExplicitData = ExplicitDynamicsDataTpl<double>;
  explicit MyModel(const ManifoldPtr &space)
      : ExplicitDynamicsModelTpl<double>(space, space->ndx()) {}

  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               ExplicitData &data) const {
    space_next().integrate(x, u, data.xnext_);
  }

  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                ExplicitData &data) const {
    space_next().Jintegrate(x, u, data.Jx_, 0);
    space_next().Jintegrate(x, u, data.Ju_, 1);
  }
};

struct MyCost : CostAbstractTpl<double> {
  using CostAbstractTpl<double>::CostAbstractTpl;
  void evaluate(const ConstVectorRef &, const ConstVectorRef &,
                CostData &data) const {
    data.value_ = 0.;
  }

  void computeGradients(const ConstVectorRef &, const ConstVectorRef &,
                        CostData &data) const {
    data.grad_.setZero();
  }

  void computeHessians(const ConstVectorRef &, const ConstVectorRef &,
                       CostData &data) const {
    data.hess_.setZero();
  }
};

#ifdef ALIGATOR_WITH_PINOCCHIO
using Manifold = aligator::SETpl<3, double>;
static const Manifold my_space;
#else
using Manifold = aligator::VectorSpaceTpl<double>;
static const Manifold my_space(6);
#endif

using StageModel = aligator::StageModelTpl<double>;
using EqualityConstraint = EqualityConstraintTpl<double>;

struct MyFixture {
  Manifold space;
  const int nu;
  const MyModel dyn_model;
  const MyCost cost;
  TrajOptProblemTpl<double> problem;

  MyFixture()
      : space(::my_space)
      , nu(space.ndx())
      , dyn_model(MyModel(space))
      , cost(MyCost(space, nu))
      , problem(space.neutral(), nu, space, cost) {
    auto stage = StageModel(cost, dyn_model);
    auto func = StateErrorResidualTpl<double>(space, nu, space.neutral());
    auto stage2 = StageModel(cost, dyn_model);
    stage2.addConstraint(func, EqualityConstraint());
    problem.addStage(stage);
    problem.addStage(stage2);
  }
};

using namespace aligator;

TEST_CASE("test_problem", "[node]") {
  MyFixture f;

  auto nu = f.nu;
  auto &space = f.space;
  const auto &stage = *f.problem.stages_[0];
  REQUIRE(stage.numPrimal() == space.ndx() + nu);
  REQUIRE(stage.numDual() == space.ndx());

  auto *p_dyn = stage.getDynamics<MyModel>();
  REQUIRE(p_dyn != nullptr);
  auto *p_cost = stage.getCost<MyCost>();
  REQUIRE(p_cost != nullptr);

  Eigen::VectorXd u0(nu);
  u0.setZero();
  auto x0 = stage.xspace_->rand();
  constexpr int nsteps = 20;
  std::vector<Eigen::VectorXd> us(nsteps, u0);

  auto xs = rollout(f.dyn_model, x0, us);
  for (std::size_t i = 0; i < xs.size(); i++) {
    REQUIRE(x0.isApprox(xs[i]));
  }

  fmt::print("{}\n", stage);

  auto stage_data = stage.createData();
  stage.evaluate(x0, u0, x0, *stage_data);
  REQUIRE(stage_data->cost_data->value_ == 0.);

  TrajOptDataTpl<double> prob_data(f.problem);
  f.problem.evaluate({x0, xs[1], xs[2]}, {u0, u0}, prob_data);
  f.problem.computeDerivatives({x0, xs[1], xs[2]}, {u0, u0}, prob_data);
}

TEST_CASE("test_workspace", "[node]") {
  using Workspace = WorkspaceTpl<double>;
  MyFixture f;
  auto nu = f.nu;
  auto space = f.space;
  Workspace workspace(f.problem);
  fmt::print("{}", workspace);
  const std::size_t nsteps = f.problem.numSteps();
  REQUIRE(workspace.nsteps == nsteps);
  REQUIRE(workspace.trial_xs.size() == nsteps + 1);

  for (std::size_t i = 0; i < nsteps; i++) {
    auto &x = workspace.trial_xs[i];
    auto &u = workspace.trial_us[i];
    REQUIRE(x.size() == space.nx());
    REQUIRE(u.size() == nu);
  }
  auto &x = workspace.trial_xs[nsteps];
  REQUIRE(x.size() == space.nx());

  ResultsTpl<double> results(f.problem);
}

TEST_CASE("test_copy", "[node]") {
  MyFixture f;

  auto copy = f.problem;
  REQUIRE(copy.getInitState() == f.problem.getInitState());

  Eigen::VectorXd state = f.problem.getInitState();

  state[0] = 0.;
  f.problem.setInitState(state);
  REQUIRE(f.problem.getInitState()[0] == 0.);

  state[0] = 1.;
  REQUIRE(f.problem.getInitState()[0] == 0.);

  copy.setInitState(state);
  REQUIRE(copy.getInitState()[0] == 1.);
  REQUIRE(f.problem.getInitState()[0] == 0.);
}

TEST_CASE("test_default_init", "[node]") {
  MyFixture f;

  Eigen::VectorXd state = f.problem.getInitState();
  REQUIRE(state == f.space.neutral());

  state[0] += 1.;
  f.problem.setInitState(state);

  auto ddp = SolverProxDDPTpl<double>();
  ddp.setup(f.problem);

  const auto nsteps = f.problem.numSteps();
  for (auto i = 0u; i < nsteps + 1; i++) {
    const auto &xs = ddp.results_.xs;
    const auto &trial_xs = ddp.workspace_.trial_xs;
    const auto &prev_xs = ddp.workspace_.prev_xs;

    REQUIRE(xs.size() == nsteps + 1);
    REQUIRE(trial_xs.size() == nsteps + 1);
    REQUIRE(prev_xs.size() == nsteps + 1);

    const auto expected = (i == 0) ? state : f.space.neutral();

    REQUIRE(xs[i] == expected);
    REQUIRE(trial_xs[i] == expected);
    REQUIRE(prev_xs[i] == expected);
  }
}

TEST_CASE("test_constant_init", "[node]") {
  MyFixture f;

  Eigen::VectorXd state = f.problem.getInitState();
  REQUIRE(state == f.space.neutral());

  state[0] += 1.;
  f.problem.setInitState(state);

  f.problem.setInitializationStrategy([](const auto &problem, auto &xs) {
    xs.assign(problem.numSteps() + 1, problem.getInitState());
  });

  auto ddp = SolverProxDDPTpl<double>();
  ddp.setup(f.problem);

  const auto nsteps = f.problem.numSteps();
  for (auto i = 0u; i < nsteps + 1; i++) {
    const auto &xs = ddp.results_.xs;
    const auto &trial_xs = ddp.workspace_.trial_xs;
    const auto &prev_xs = ddp.workspace_.prev_xs;

    REQUIRE(xs.size() == nsteps + 1);
    REQUIRE(trial_xs.size() == nsteps + 1);
    REQUIRE(prev_xs.size() == nsteps + 1);

    REQUIRE(xs[i] == state);
    REQUIRE(trial_xs[i] == state);
    REQUIRE(prev_xs[i] == state);
  }
}
