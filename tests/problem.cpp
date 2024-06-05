#include "aligator/core/traj-opt-problem.hpp"
#include "aligator/solvers/proxddp/results.hpp"
#include "aligator/solvers/proxddp/workspace.hpp"
#include "aligator/core/explicit-dynamics.hpp"
#include "aligator/core/cost-abstract.hpp"
#include "aligator/utils/rollout.hpp"
#include "aligator/modelling/state-error.hpp"
#include <proxsuite-nlp/modelling/spaces/pinocchio-groups.hpp>

#include <boost/test/unit_test.hpp>

#include <fmt/core.h>
#include <fmt/ostream.h>

using namespace aligator;

/// @brief    Addition dynamics.
/// @details  It maps \f$(x,u)\f$ to \f$ x + u \f$.
struct MyModel : ExplicitDynamicsModelTpl<double> {
  using Manifold = ManifoldAbstractTpl<double>;
  using ExplicitData = ExplicitDynamicsDataTpl<double>;
  explicit MyModel(const shared_ptr<Manifold> &space)
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

using Manifold = proxsuite::nlp::SETpl<3, double>;
using StageModel = aligator::StageModelTpl<double>;
using EqualityConstraint = proxsuite::nlp::EqualityConstraintTpl<double>;

struct MyFixture {
  shared_ptr<Manifold> space;
  const int nu;
  const shared_ptr<MyModel> dyn_model;
  const shared_ptr<MyCost> cost;
  TrajOptProblemTpl<double> problem;

  MyFixture()
      : space(std::make_shared<Manifold>()), nu(space->ndx()),
        dyn_model(std::make_shared<MyModel>(space)),
        cost(std::make_shared<MyCost>(space, nu)),
        problem(space->neutral(), nu, space, cost) {
    auto stage = std::make_shared<StageModel>(cost, dyn_model);
    auto func = std::make_shared<StateErrorResidualTpl<double>>(
        space, nu, space->neutral());
    auto stage2 = stage->clone();
    stage2->addConstraint(func, std::make_shared<EqualityConstraint>());
    problem.addStage(stage);
    problem.addStage(stage2);
  }
};

BOOST_AUTO_TEST_SUITE(node)

using namespace aligator;

BOOST_AUTO_TEST_CASE(test_problem) {
  MyFixture f;

  auto nu = f.nu;
  auto &space = *f.space;
  auto &stage = *f.problem.stages_[0];
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
