#include "proxddp/core/stage-model.hpp"
#include "proxddp/core/solver-workspace.hpp"
#include "proxddp/utils.hpp"

#include "generate-problem.hpp"
#include <proxnlp/modelling/spaces/vector-space.hpp>

#include <boost/test/unit_test.hpp>

#include <fmt/core.h>
#include <fmt/ostream.h>


BOOST_AUTO_TEST_SUITE(node)

using namespace proxddp;

using Scalar = double;

using Manifold = proxnlp::VectorSpaceTpl<double>;
using StageModel = proxddp::StageModelTpl<double>;

constexpr int NX = 3;
Manifold space(NX);
int NU = space.ndx();
MyModel dyn_model(space);
MyCost cost(NX, NU);
StageModel stage(space, NU, cost, dyn_model);
ShootingProblemTpl<double> problem { {stage, stage} };


BOOST_AUTO_TEST_CASE(test_problem)
{
  BOOST_CHECK_EQUAL(stage.numPrimal(), NX + NU);
  BOOST_CHECK_EQUAL(stage.numDual(), NX);

  Eigen::VectorXd u0(NU);
  u0.setZero();
  auto x0 = stage.xspace1_.rand();
  constexpr int nsteps = 20;
  std::vector<Eigen::VectorXd> us(nsteps, u0);

  auto xs = rollout(dyn_model, x0, us);
  for (std::size_t i = 0; i < xs.size(); i++)
  {
    BOOST_CHECK(x0.isApprox(xs[i]));
  }

  auto stage_data = stage.createData();
  stage.evaluate(x0, u0, x0, *stage_data);
  BOOST_CHECK_EQUAL(stage_data->cost_data->value_, 0.);

  auto prob_data = problem.createData();
  problem.evaluate({x0, xs[1], xs[2]}, {u0, u0}, *prob_data);
  problem.computeDerivatives({x0, xs[1], xs[2]}, {u0, u0}, *prob_data);
}


BOOST_AUTO_TEST_CASE(test_workspace)
{
  using Workspace = WorkspaceTpl<double>;
  Workspace workspace(problem);
  const std::size_t nsteps = problem.numSteps();
  BOOST_CHECK_EQUAL(workspace.trial_xs_.size(), nsteps);

  for (std::size_t i = 0; i < nsteps; i++)
  {
    auto& x = workspace.trial_xs_[i];
    auto& u = workspace.trial_us_[i];
    fmt::print("{} << x{:d}\n", x, i);
    BOOST_CHECK_EQUAL(x.size(), NX);
    BOOST_CHECK_EQUAL(u.size(), NU);
  }
  auto& x = workspace.trial_xs_[nsteps];
  BOOST_CHECK_EQUAL(x.size(), NX);

}


BOOST_AUTO_TEST_SUITE_END()
