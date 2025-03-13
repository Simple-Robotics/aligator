#include "aligator/core/traj-opt-problem.hpp"
#include "aligator/core/traj-opt-data.hpp"
#include "aligator/core/stage-data.hpp"
#include "aligator/solvers/proxddp/solver-proxddp.hpp"
#include "aligator/solvers/proxddp/results.hpp"
#include "aligator/solvers/proxddp/workspace.hpp"
#include "aligator/core/explicit-dynamics.hpp"
#include "aligator/core/cost-abstract.hpp"
#include "aligator/utils/rollout.hpp"
#ifdef PROXSUITE_NLP_WITH_PINOCCHIO
#include <proxsuite-nlp/modelling/spaces/pinocchio-groups.hpp>
#else
#include <proxsuite-nlp/modelling/spaces/vector-space.hpp>
#endif
#include <proxsuite-nlp/third-party/polymorphic_cxx14.hpp>
#include <boost/test/unit_test.hpp>

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

#ifdef PROXSUITE_NLP_WITH_PINOCCHIO
using Manifold = proxsuite::nlp::SETpl<3, double>;
static const Manifold my_space;
#else
using Manifold = proxsuite::nlp::VectorSpaceTpl<double>;
static const Manifold my_space(6);
#endif

using StageModel = aligator::StageModelTpl<double>;
using EqualityConstraint = proxsuite::nlp::EqualityConstraintTpl<double>;

struct MyFixture {
  Manifold space;
  const int nu;
  const MyModel dyn_model;
  const MyCost cost;
  TrajOptProblemTpl<double> problem;

  MyFixture()
      : space(::my_space), nu(space.ndx()), dyn_model(MyModel(space)),
        cost(MyCost(space, nu)), problem(space.neutral(), nu, space, cost) {
    auto stage = StageModel(cost, dyn_model);
    auto func = StateErrorResidualTpl<double>(space, nu, space.neutral());
    auto stage2 = StageModel(cost, dyn_model);
    stage2.addConstraint(func, EqualityConstraint());
    problem.addStage(stage);
    problem.addStage(stage2);
  }
};

BOOST_AUTO_TEST_SUITE(node)

using namespace aligator;

BOOST_AUTO_TEST_CASE(test_problem) {
  MyFixture f;

  auto nu = f.nu;
  auto &space = f.space;
  const auto &stage = *f.problem.stages_[0];
  BOOST_CHECK_EQUAL(stage.numPrimal(), space.ndx() + nu);
  BOOST_CHECK_EQUAL(stage.numDual(), space.ndx());

  auto *p_dyn = stage.getDynamics<MyModel>();
  BOOST_CHECK(p_dyn);
  auto *p_cost = stage.getCost<MyCost>();
  BOOST_CHECK(p_cost);

  Eigen::VectorXd u0(nu);
  u0.setZero();
  auto x0 = stage.xspace_->rand();
  constexpr int nsteps = 20;
  std::vector<Eigen::VectorXd> us(nsteps, u0);

  auto xs = rollout(f.dyn_model, x0, us);
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
  auto space = f.space;
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

BOOST_AUTO_TEST_CASE(test_copy) {
  MyFixture f;

  auto copy = f.problem;
  BOOST_CHECK_EQUAL(copy.getInitState(), f.problem.getInitState());

  Eigen::VectorXd state = f.problem.getInitState();

  state[0] = 0.;
  f.problem.setInitState(state);
  BOOST_CHECK_EQUAL(f.problem.getInitState()[0], 0.);

  state[0] = 1.;
  BOOST_CHECK_EQUAL(f.problem.getInitState()[0], 0.);

  copy.setInitState(state);
  BOOST_CHECK_EQUAL(copy.getInitState()[0], 1.);
  BOOST_CHECK_EQUAL(f.problem.getInitState()[0], 0.);
}

BOOST_AUTO_TEST_CASE(test_default_init) {
  MyFixture f;

  Eigen::VectorXd state = f.problem.getInitState();
  BOOST_CHECK_EQUAL(state, f.space.neutral());

  state[0] += 1.;
  f.problem.setInitState(state);

  auto ddp = SolverProxDDPTpl<double>();
  ddp.setup(f.problem);

  const auto nsteps = f.problem.numSteps();
  for (auto i = 0u; i < nsteps + 1; i++) {
    const auto &xs = ddp.results_.xs;
    const auto &trial_xs = ddp.workspace_.trial_xs;
    const auto &prev_xs = ddp.workspace_.prev_xs;

    BOOST_REQUIRE_EQUAL(xs.size(), nsteps + 1);
    BOOST_REQUIRE_EQUAL(trial_xs.size(), nsteps + 1);
    BOOST_REQUIRE_EQUAL(prev_xs.size(), nsteps + 1);

    const auto expected = (i == 0) ? state : f.space.neutral();

    BOOST_CHECK_EQUAL(xs[i], expected);
    BOOST_CHECK_EQUAL(trial_xs[i], expected);
    BOOST_CHECK_EQUAL(prev_xs[i], expected);
  }
}

BOOST_AUTO_TEST_CASE(test_constant_init) {
  MyFixture f;

  Eigen::VectorXd state = f.problem.getInitState();
  BOOST_CHECK_EQUAL(state, f.space.neutral());

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

    BOOST_REQUIRE_EQUAL(xs.size(), nsteps + 1);
    BOOST_REQUIRE_EQUAL(trial_xs.size(), nsteps + 1);
    BOOST_REQUIRE_EQUAL(prev_xs.size(), nsteps + 1);

    BOOST_CHECK_EQUAL(xs[i], state);
    BOOST_CHECK_EQUAL(trial_xs[i], state);
    BOOST_CHECK_EQUAL(prev_xs[i], state);
  }
}

BOOST_AUTO_TEST_SUITE_END()
