#include "proxddp/core/solver-proxddp.hpp"
#include "proxddp/utils/rollout.hpp"

#include "generate-problem.hpp"
#include <proxnlp/modelling/spaces/vector-space.hpp>

#include <boost/test/unit_test.hpp>

#include <fmt/core.h>
#include <fmt/ostream.h>

BOOST_AUTO_TEST_SUITE(node)

using namespace proxddp;

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

BOOST_AUTO_TEST_SUITE_END()
