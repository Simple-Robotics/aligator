#include "aligator/core/traj-opt-problem.hpp"
#include "aligator/solvers/proxddp/results.hpp"
#include "aligator/solvers/proxddp/workspace.hpp"
#include "aligator/utils/rollout.hpp"

#include "generate-problem.hpp"
#include <proxsuite-nlp/modelling/spaces/vector-space.hpp>

#include <boost/test/unit_test.hpp>

#include <fmt/core.h>
#include <fmt/ostream.h>

BOOST_AUTO_TEST_SUITE(node)

using namespace aligator;

BOOST_AUTO_TEST_CASE(test_problem) {
  MyFixture f;

  auto nu = f.nu;
  auto &space = *f.space;
  auto &stage = *f.stage;
  BOOST_CHECK_EQUAL(stage.numPrimal(), space.ndx() + nu);
  BOOST_CHECK_EQUAL(stage.numDual(), space.ndx());

  Eigen::VectorXd u0(nu);
  u0.setZero();
  auto x0 = stage.xspace_->rand();
  constexpr int nsteps = 20;
  std::vector<Eigen::VectorXd> us(nsteps, u0);

  auto xs = rollout(*f.dyn_model, x0, us);
  for (std::size_t i = 0; i < xs.size(); i++) {
    BOOST_CHECK(x0.isApprox(xs[i]));
  }

  fmt::print("{}\n", stage);

  auto stage_data = stage.createData();
  stage.evaluate(x0, u0, x0, *stage_data);
  BOOST_CHECK_EQUAL(stage_data->cost_data->value_, 0.);

  TrajOptDataTpl<double> prob_data(f.problem);
  f.problem.evaluate({x0, xs[1], xs[2]}, {u0, u0}, prob_data);
  f.problem.computeDerivatives({x0, xs[1], xs[2]}, {u0, u0}, prob_data);
}

BOOST_AUTO_TEST_CASE(test_workspace) {
  using Workspace = WorkspaceTpl<double>;
  MyFixture f;
  auto nu = f.nu;
  auto space = *f.space;
  Workspace workspace(f.problem);
  fmt::print("{}", workspace);
  const std::size_t nsteps = f.problem.numSteps();
  BOOST_CHECK_EQUAL(workspace.nsteps, nsteps);
  BOOST_CHECK_EQUAL(workspace.trial_xs.size(), nsteps + 1);

  for (std::size_t i = 0; i < nsteps; i++) {
    auto &x = workspace.trial_xs[i];
    auto &u = workspace.trial_us[i];
    BOOST_CHECK_EQUAL(x.size(), space.nx());
    BOOST_CHECK_EQUAL(u.size(), nu);
  }
  auto &x = workspace.trial_xs[nsteps];
  BOOST_CHECK_EQUAL(x.size(), space.nx());

  ResultsTpl<double> results(f.problem);
}

namespace {

struct MockModel : MyModel {
  using Scalar = double;
  using CommonModelBuilderContainer = CommonModelBuilderContainerTpl<Scalar>;

  MockModel() : MyModel(std::make_shared<::Manifold>()) {}

  void configure(
      ALIGATOR_MAYBE_UNUSED CommonModelBuilderContainer &container) const {
    configure_called = true;
  }

  mutable bool configure_called = false;
};

struct MockUnaryFunction : UnaryFunctionTpl<double> {
  MockUnaryFunction() : UnaryFunctionTpl<double>(6, 6, 6) {}

  void configure(ALIGATOR_MAYBE_UNUSED CommonModelBuilderContainer &container)
      const override {
    configure_called = true;
  }

  void evaluate(const ConstVectorRef &, Data &) const override {}
  void computeJacobians(const ConstVectorRef &, Data &) const override {}

  mutable bool configure_called = false;
};

struct MockCost : MyCost {
  using Scalar = double;
  using CommonModelBuilderContainer = CommonModelBuilderContainerTpl<Scalar>;

  MockCost() : MyCost(std::make_shared<::Manifold>(), 6) {}

  void configure(
      ALIGATOR_MAYBE_UNUSED CommonModelBuilderContainer &container) const {
    configure_called = true;
  }

  mutable bool configure_called = false;
};

} // namespace

/// Test that configure is well called for:
/// - Cost and constraints in each stage
/// - Final cost and constraints
/// - Initial constraint
BOOST_AUTO_TEST_CASE(test_problem_configure) {
  auto model1 = std::make_shared<MockModel>();
  auto model2 = std::make_shared<MockModel>();
  auto init_constr = std::make_shared<MockUnaryFunction>();
  auto final_constr = std::make_shared<MockModel>();

  auto cost1 = std::make_shared<MockCost>();
  auto cost2 = std::make_shared<MockCost>();
  auto final_cost = std::make_shared<MockCost>();
  auto stage1 = std::make_shared<StageModel>(cost1, model1);
  auto stage2 = std::make_shared<StageModel>(cost2, model2);

  TrajOptProblemTpl<double> problem(init_constr, {stage1, stage2}, final_cost);
  using EqualitySet = proxsuite::nlp::EqualityConstraint<double>;
  problem.addTerminalConstraint(StageConstraintTpl<double>{
      final_constr, std::make_shared<EqualitySet>()});

  problem.configure();
  BOOST_CHECK(model1->configure_called);
  BOOST_CHECK(model2->configure_called);
  BOOST_CHECK(init_constr->configure_called);
  BOOST_CHECK(final_constr->configure_called);
  BOOST_CHECK(cost1->configure_called);
  BOOST_CHECK(cost2->configure_called);
  BOOST_CHECK(final_cost->configure_called);
}

BOOST_AUTO_TEST_SUITE_END()
