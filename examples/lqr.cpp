/// @file
/// @brief Linear-quadratic regulator

#include "proxddp/core/solver-proxddp.hpp"
#include "proxddp/utils/rollout.hpp"
#include "proxddp/modelling/quad-costs.hpp"

#include "proxddp/fddp/solver-fddp.hpp"

#include <proxnlp/modelling/constraints/negative-orthant.hpp>

#include "proxddp/modelling/linear-discrete-dynamics.hpp"
#include "proxddp/modelling/control-box-function.hpp"

using namespace proxddp;

constexpr double TOL = 1e-7;

int main() {

  const int dim = 2;
  const int nu = 2;
  Eigen::MatrixXd A(dim, dim);
  Eigen::MatrixXd B(dim, nu);
  Eigen::VectorXd c_(dim);
  A.setIdentity();
  B << -0.6, 0.3, 0., 1.;
  c_ << 0.1, 0.;

  Eigen::MatrixXd w_x(dim, dim), w_u(nu, nu);
  w_x.setIdentity();
  w_u.setIdentity();
  w_x(0, 0) = 2.1;
  w_u *= 1e-2;

  using dynamics::LinearDiscreteDynamicsTpl;
  auto dynptr = std::make_shared<LinearDiscreteDynamicsTpl<double>>(A, B, c_);
  auto &dynamics = *dynptr;
  fmt::print("matrix A:\n{}\n", dynamics.A_);
  fmt::print("matrix B:\n{}\n", dynamics.B_);
  fmt::print("drift  c:\n{}\n", dynamics.c_);
  auto spaceptr = dynamics.next_state_;

  auto rcost = std::make_shared<QuadraticCostTpl<double>>(w_x, w_u);

  // Define stage

  double u_bound = 0.2;
  auto stage =
      std::make_shared<StageModelTpl<double>>(spaceptr, nu, rcost, dynptr);
  auto ctrl_bounds_fun = std::make_shared<ControlBoxFunctionTpl<double>>(
      dim, nu, -u_bound, u_bound);

  const bool HAS_CONTROL_BOUNDS = false;

  if (HAS_CONTROL_BOUNDS) {
    fmt::print("Adding control bounds.\n");
    fmt::print("control box fun has bounds:\n{} max\n{} min\n",
               ctrl_bounds_fun->umax_, ctrl_bounds_fun->umin_);
    using InequalitySet = proxnlp::NegativeOrthant<double>;
    stage->addConstraint(ctrl_bounds_fun, std::make_shared<InequalitySet>());
  }

  Eigen::VectorXd x0(2);
  x0 << 1., -0.1;

  auto &term_cost = rcost;
  TrajOptProblemTpl<double> problem(x0, nu, spaceptr, term_cost);

  std::size_t nsteps = 10;

  std::vector<Eigen::VectorXd> us_init;
  for (std::size_t i = 0; i < nsteps; i++) {
    us_init.push_back(Eigen::VectorXd::Random(nu));
    problem.addStage(stage);
  }

  auto xs_init = rollout(dynamics, x0, us_init);

  const double mu_init = 1e-5;
  const double rho_init = 1e-8;

  SolverProxDDP<double> solver(TOL, mu_init, rho_init);
  solver.verbose_ = VerboseLevel::VERBOSE;

  solver.setup(problem);
  solver.run(problem, xs_init, us_init);
  const auto &results = solver.getResults();

  std::string line_ = "";
  for (std::size_t i = 0; i < 20; i++) {
    line_.append("=");
  }
  line_.append("\n");
  for (std::size_t i = 0; i < nsteps + 1; i++) {
    // fmt::print("x[{:d}] = {}\n", i, results.xs_[i].transpose());
    fmt::print("x[{:d}] = {}\n", i, results.xs_[i].transpose());
  }
  for (std::size_t i = 0; i < nsteps; i++) {
    // fmt::print("u[{:d}] = {}\n", i, results.us_[i].transpose());
    fmt::print("u[{:d}] = {}\n", i, results.us_[i].transpose());
  }

  {
    fmt::print("TEST FDDP\n");
    SolverFDDP<double> fddp(TOL, 1e-10, VerboseLevel::VERBOSE);
    fddp.MAX_ITERS = 20;
    fddp.setup(problem);
    fddp.run(problem, xs_init, us_init);
    fmt::print("FDDP done.\n");
    const ResultsFDDP<double> &res_fddp = fddp.getResults();
    for (std::size_t i = 0; i < nsteps + 1; i++) {
      fmt::print("x[{:d}] = {}\n", i, res_fddp.xs_[i].transpose());
    }
  }

  return 0;
}
