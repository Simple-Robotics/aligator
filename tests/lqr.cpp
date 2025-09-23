#include "aligator/modelling/linear-discrete-dynamics.hpp"
#include "aligator/modelling/costs/quad-costs.hpp"
#include "aligator/solvers/proxddp/solver-proxddp.hpp"

#include <aligator/fmt-eigen.hpp>

#include <catch2/catch_test_macros.hpp>
#include <random>

using namespace aligator;

using LinearDynamics = dynamics::LinearDiscreteDynamicsTpl<double>;
using QuadraticCost = QuadraticCostTpl<double>;
using context::CostAbstract;
using context::SolverProxDDP;
using context::StageModel;
using context::TrajOptProblem;

using Eigen::MatrixXd;
using Eigen::VectorXd;

static std::mt19937_64 urng{42};
struct NormalGen {
  double operator()() const { return norm(urng); }
  mutable std::normal_distribution<double> norm;
};

TEST_CASE("lqr_proxddp") {
  const size_t nsteps = 100;
  const auto nx = 4;
  const auto nu = 2;

  NormalGen norm_gen;
  MatrixXd A;
  // clang-format off
  A.setIdentity(nx, nx);
  A.bottomRightCorner<2, 2>() = MatrixXd::NullaryExpr(2, 2, norm_gen);
  MatrixXd B = MatrixXd::NullaryExpr(nx, nu, norm_gen);
  // clang-format on

  VectorXd x0 = VectorXd::NullaryExpr(nx, norm_gen);

  auto dyn_model = LinearDynamics(A, B, VectorXd::Zero(nx));
  MatrixXd Q = MatrixXd::NullaryExpr(nx, nx, norm_gen);
  Q = Q.transpose() * Q;
  VectorXd q = VectorXd::NullaryExpr(nx, norm_gen);

  MatrixXd R = MatrixXd::NullaryExpr(nu, nu, norm_gen);
  R = R.transpose() * R;
  VectorXd r = VectorXd::Zero(nu);

  QuadraticCost cost = QuadraticCost(Q, R, q, r);
  QuadraticCost term_cost = QuadraticCost(Q * 10., MatrixXd());
  assert(term_cost.nu == 0);

  auto stage = StageModel(cost, dyn_model);

  std::vector<xyz::polymorphic<StageModel>> stages(nsteps, stage);
  TrajOptProblem problem(x0, stages, term_cost);

  double tol = 1e-6;
  double mu_init = 1e-8;
  SolverProxDDP ddp(tol, mu_init);
  ddp.rollout_type_ = RolloutType::LINEAR;
  ddp.max_iters = 2;
  ddp.verbose_ = VERBOSE;

  ddp.setup(problem);
  bool conv = ddp.run(problem);
  REQUIRE(conv);
  REQUIRE(ddp.results_.num_iters == 1);

  fmt::println("{}", ddp.results_);
}
