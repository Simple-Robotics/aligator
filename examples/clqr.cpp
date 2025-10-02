#include "aligator/modelling/linear-discrete-dynamics.hpp"
#include "aligator/modelling/linear-function.hpp"
#include "aligator/modelling/costs/quad-costs.hpp"
#include "aligator/modelling/state-error.hpp"
#include "aligator/solvers/proxddp/solver-proxddp.hpp"

#include "aligator/modelling/constraints.hpp"
#include <aligator/fmt-eigen.hpp>
#include <iostream>
#include <random>

using namespace aligator;

using Space = VectorSpaceTpl<double>;
using LinearDynamics = dynamics::LinearDiscreteDynamicsTpl<double>;
using LinearFunction = LinearFunctionTpl<double>;
using BoxConstraint = BoxConstraintTpl<double>;
using EqualityConstraint = EqualityConstraintTpl<double>;
using QuadraticCost = QuadraticCostTpl<double>;
using context::CostAbstract;
using context::MatrixXs;
using context::StageModel;
using context::TrajOptProblem;
using context::VectorXs;

static std::mt19937_64 urng{42};
struct NormalGen {
  double operator()() const { return norm(urng); }
  mutable std::normal_distribution<double> norm;
};

int main() {
  std::srand(42);
  const size_t nsteps = 100;
  const auto nx = 4;
  const auto nu = 2;
  const auto space = Space(nx);

  NormalGen norm_gen;
  MatrixXs A;
  // clang-format off
  A.setIdentity(nx, nx);
  A.bottomRightCorner<2, 2>() = MatrixXs::NullaryExpr(2, 2, norm_gen);
  MatrixXs B = MatrixXs::NullaryExpr(nx, nu, norm_gen);
  // clang-format on

  VectorXs x0 = VectorXs::NullaryExpr(nx, norm_gen);

  auto dyn_model = LinearDynamics(A, B, VectorXs::Zero(nx));

  MatrixXs Q = MatrixXs::NullaryExpr(nx, nx, norm_gen);
  Q = Q.transpose() * Q;
  VectorXs q = VectorXs::Zero(nx);

  MatrixXs R = MatrixXs::NullaryExpr(nu, nu, norm_gen);
  R = R.transpose() * R;
  VectorXs r = VectorXs::Zero(nu);

  QuadraticCost cost = QuadraticCost(Q, R, q, r);
  QuadraticCost term_cost = QuadraticCost(Q * 10., MatrixXs());
  assert(term_cost.nu == 0);

  double ctrlUpperBound = 0.3;
  auto stage = StageModel(cost, dyn_model);
  {
    auto box = BoxConstraint(-ctrlUpperBound * VectorXs::Ones(nu),
                             ctrlUpperBound * VectorXs::Ones(nu));
    auto u0 = VectorXs::Zero(nu);
    auto func = ControlErrorResidualTpl<double>(nx, u0);
    stage.addConstraint(func, box);
  }

  std::vector<xyz::polymorphic<StageModel>> stages(nsteps, stage);
  TrajOptProblem problem(x0, stages, term_cost);

  bool terminal = false;
  if (terminal) {
    StateErrorResidualTpl<double> func(space, nu, VectorXs::Ones(nx));
    problem.addTerminalConstraint(func, EqualityConstraint());
  }

  const double tol = 1e-6;
  const double mu_init = 1e-6;
  SolverProxDDPTpl<double> solver(tol, mu_init);
  solver.max_iters = 10;
  solver.verbose_ = VERBOSE;
  solver.linear_solver_choice = LQSolverChoice::PARALLEL;
  solver.force_initial_condition_ = true;
  solver.rollout_type_ = RolloutType::LINEAR;
  solver.setNumThreads(4);

  solver.setup(problem);
  const bool conv = solver.run(problem);
  (void)conv;

  auto us = solver.results_.us;
  for (std::size_t i = 0; i < us.size(); i++) {
    fmt::print("us[{:02d}] = {}\n", i, us[i].transpose());
  }
  auto xs = solver.results_.xs;
  for (std::size_t i = 0; i < xs.size(); i++) {
    fmt::print("xs[{:02d}] = {}\n", i, xs[i].transpose());
  }

  std::cout << solver.results_ << std::endl;

  assert(conv);
}
