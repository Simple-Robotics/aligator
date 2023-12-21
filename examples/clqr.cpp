#include "aligator/modelling/linear-discrete-dynamics.hpp"
#include "aligator/modelling/linear-function.hpp"
#include "aligator/modelling/costs/quad-costs.hpp"
#include "aligator/modelling/state-error.hpp"
#include "aligator/solvers/proxddp/solver-proxddp.hpp"

#include <proxsuite-nlp/modelling/constraints.hpp>
#include <iostream>

using namespace aligator;

using Space = proxsuite::nlp::VectorSpaceTpl<double>;
using LinearDynamics = dynamics::LinearDiscreteDynamicsTpl<double>;
using LinearFunction = LinearFunctionTpl<double>;
using BoxConstraint = proxsuite::nlp::BoxConstraintTpl<double>;
using Equality = proxsuite::nlp::EqualityConstraint<double>;
using QuadraticCost = QuadraticCostTpl<double>;
using context::CostBase;
using context::StageModel;
using context::TrajOptProblem;

using Eigen::MatrixXd;
using Eigen::VectorXd;

static std::mt19937_64 urng{42};
struct NormalGen {
  double operator()() const { return norm(urng); }
  mutable std::normal_distribution<double> norm;
};

int main() {
  const size_t nsteps = 100;
  const auto nx = 4;
  const auto nu = 2;
  const auto space = std::make_shared<Space>(nx);

  NormalGen norm_gen;
  MatrixXd A;
  // clang-format off
  A.setIdentity(nx, nx);
  A.bottomRightCorner<2, 2>() = MatrixXd::NullaryExpr(2, 2, norm_gen);
  MatrixXd B = MatrixXd::NullaryExpr(nx, nu, norm_gen);
  // clang-format on

  VectorXd x0 = VectorXd::NullaryExpr(nx, norm_gen);

  auto dyn_model = std::make_shared<LinearDynamics>(A, B, VectorXd::Zero(nx));
  shared_ptr<CostBase> cost, term_cost;
  {
    MatrixXd Q = MatrixXd::NullaryExpr(nx, nx, norm_gen);
    Q = Q.transpose() * Q;
    VectorXd q = VectorXd::Zero(nx);

    MatrixXd R = MatrixXd::NullaryExpr(nu, nu, norm_gen);
    R = R.transpose() * R;
    VectorXd r = VectorXd::Zero(nu);

    cost = std::make_shared<QuadraticCost>(Q, R, q, r);
    term_cost = std::make_shared<QuadraticCost>(Q * 10., MatrixXd());
    assert(term_cost->nu == 0);
  }

  auto stage = std::make_shared<StageModel>(cost, dyn_model);
  {
    double ub = 1.0;
    auto box = std::make_shared<BoxConstraint>(-ub * VectorXd::Ones(nu),
                                               ub * VectorXd::Ones(nu));
    auto func = std::make_shared<ControlErrorResidualTpl<double>>(
        nx, VectorXd::Zero(nu));
    stage->addConstraint(func, box);
  }

  std::vector<decltype(stage)> stages(nsteps);
  std::fill(stages.begin(), stages.end(), stage);
  TrajOptProblem problem(x0, stages, term_cost);

  double tol = 1e-6;
  SolverProxDDP<double> ddp(tol, 0.01);
  ddp.max_iters = 10;
  ddp.verbose_ = VERBOSE;

  ddp.setup(problem);
  bool conv = ddp.run(problem);
  (void)conv;
  assert(conv);

  std::cout << ddp.results_ << std::endl;
}
