/// @file
/// @brief Linear-quadratic regulator

#include "proxddp/core/shooting-problem.hpp"
#include "proxddp/utils.hpp"
#include "proxddp/modelling/quad-costs.hpp"
#include "proxddp/core/solver-proxddp.hpp"

#include <proxnlp/modelling/constraints/negative-orthant.hpp>

#include "proxddp/modelling/linear-discrete-dynamics.hpp"
#include "proxddp/modelling/box-constraints.hpp"

using namespace proxddp;


constexpr double TOL = 1e-7;

int main()
{

  const int dim = 2;
  const int nu = 1;
  Eigen::MatrixXd A(dim, dim);
  Eigen::MatrixXd B(dim, nu);
  Eigen::VectorXd c_(dim);
  A.setIdentity();
  B << -0.6, 0.3;
  c_ << 0.1, 0.;

  Eigen::MatrixXd w_x(dim, dim), w_u(nu, nu);
  w_x.setIdentity();
  w_u.setIdentity();
  w_x(0, 0) = 2.1;
  w_u *= 1e-2;

  auto dynptr = std::make_shared<LinearDiscreteDynamics<double>>(A, B, c_);
  auto& dynamics = *dynptr;
  fmt::print("matrix A:\n{}\n", dynamics.A_);
  fmt::print("matrix B:\n{}\n", dynamics.B_);
  fmt::print("drift  c:\n{}\n", dynamics.c_);
  const auto& space = dynamics.out_space();

  QuadraticCost<double> rcost(w_x, w_u);

  // Define stage

  double u_bound = 0.2;
  StageModelTpl<double> stage(space, nu, rcost, dynptr);
  auto ctrl_bounds_fun = std::make_shared<ControlBoxFunction<double>>(dim, nu, -u_bound, u_bound);

  const bool HAS_CONTROL_BOUNDS = true;

  if (HAS_CONTROL_BOUNDS)
  {
    fmt::print("Adding control bounds.\n");
    fmt::print("control box fun has bounds:\n{} max\n{} min\n", ctrl_bounds_fun->umax_, ctrl_bounds_fun->umin_);
    auto ctrl_bounds_cstr = std::make_shared<StageConstraintTpl<double>>(
      ctrl_bounds_fun,
      std::make_shared<proxnlp::NegativeOrthant<double>>());
    stage.addConstraint(ctrl_bounds_cstr);
  }

  auto x0 = space.rand();
  x0 << 1., -0.1;
  auto term_cost = std::make_shared<decltype(rcost)>(rcost);
  ShootingProblemTpl<double> problem(x0, nu, space, term_cost);

  std::size_t nsteps = 10;

  std::vector<Eigen::VectorXd> us;
  for (std::size_t i = 0; i < nsteps; i++)
  {
    us.push_back(Eigen::VectorXd::Random(nu));
    problem.addStage(*stage.clone());
  }

  auto xs = rollout(dynamics, x0, us);
  fmt::print("Initial traj.:\n");
  for(std::size_t i = 0; i <= nsteps; i++)
  {
    fmt::print("x[{:d}] = {}\n", i, xs[i].transpose());
  }

  const double mu_init = 1e-5;
  const double rho_init = 1e-8;

  SolverProxDDP<double> solver(TOL, mu_init, rho_init);

  WorkspaceTpl<double> workspace(problem);
  ResultsTpl<double> results(problem);
  assert(results.xs_.size() == nsteps + 1);
  assert(results.us_.size() == nsteps);

  solver.run(problem, workspace, results, xs, us);

  std::string line_ = "";
  for (std::size_t i = 0; i < 20; i++)
  { line_.append("="); }
  line_.append("\n");
  for (std::size_t i = 0; i < nsteps + 1; i++)
  {
    // fmt::print("x[{:d}] = {}\n", i, results.xs_[i].transpose());
    fmt::print("x[{:d}] = {}\n", i, workspace.trial_xs_[i].transpose());
  }
  for (std::size_t i = 0; i < nsteps; i++)
  {
    // fmt::print("u[{:d}] = {}\n", i, results.us_[i].transpose());
    fmt::print("u[{:d}] = {}\n", i, workspace.trial_us_[i].transpose());
  }

  return 0;
}
