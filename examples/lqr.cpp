/// @file
/// @brief Linear-quadratic regulator

#include "proxddp/core/solver-proxddp.hpp"
#include "proxddp/utils/rollout.hpp"
#include "proxddp/modelling/quad-costs.hpp"

#include "proxddp/fddp/solver-fddp.hpp"

#include <proxnlp/modelling/constraints/negative-orthant.hpp>

#include "proxddp/modelling/linear-discrete-dynamics.hpp"
#include "proxddp/modelling/control-box-function.hpp"

#include <benchmark/benchmark.h>
#include <iostream>

using namespace proxddp;

constexpr double TOL = 1e-7;

void define_problem(shared_ptr<TrajOptProblemTpl<double>> &problemptr) {

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
  w_x(0, 0) = 2.;
  w_u *= 1e-2;

  using dynamics::LinearDiscreteDynamicsTpl;
  auto dynptr = std::make_shared<LinearDiscreteDynamicsTpl<double>>(A, B, c_);
  auto &dynamics = *dynptr;
  auto spaceptr = dynamics.space_next_;

  auto rcost = std::make_shared<QuadraticCostTpl<double>>(w_x, w_u);

  // Define stage

  double u_bound = 0.2;
  auto stage =
      std::make_shared<StageModelTpl<double>>(spaceptr, nu, rcost, dynptr);
  auto ctrl_bounds_fun = std::make_shared<ControlBoxFunctionTpl<double>>(
      dim, nu, -u_bound, u_bound);

  const bool HAS_CONTROL_BOUNDS = false;

  if (HAS_CONTROL_BOUNDS) {
    using InequalitySet = proxnlp::NegativeOrthant<double>;
    stage->addConstraint(ctrl_bounds_fun, std::make_shared<InequalitySet>());
  }

  Eigen::VectorXd x0(2);
  x0 << 1., -0.1;

  auto &term_cost = rcost;
  problemptr =
      std::make_shared<TrajOptProblemTpl<double>>(x0, nu, spaceptr, term_cost);

  std::size_t nsteps = 10;

  for (std::size_t i = 0; i < nsteps; i++) {
    problemptr->addStage(stage);
  }
}

void BM_lqr(benchmark::State &state, const TrajOptProblemTpl<double> &problem,
            bool run_fddp) {

  for (auto _ : state) {

    const auto &dynamics = problem.stages_[0]->dyn_model();
    const auto &x0 = problem.getInitState();
    std::vector<Eigen::VectorXd> us_init;
    us_default_init(problem, us_init);
    const auto xs_init = rollout(dynamics, x0, us_init);

    auto verbose = VerboseLevel::QUIET;

    const std::size_t max_iters = 4;
    if (!run_fddp) {
      const double mu_init = 1e-7;
      const double rho_init = 0.;

      SolverProxDDP<double> solver(TOL, mu_init, rho_init, max_iters, verbose);

      solver.setup(problem);
      solver.run(problem, xs_init, us_init);
      const auto &results = solver.getResults();
      if (!results.conv) {
        proxddp_runtime_error(fmt::format(
            "Solver did not converge ({:d} iters).\n", results.num_iters));
      }
    }
    if (run_fddp) {
      SolverFDDP<double> fddp(TOL, verbose);
      fddp.MAX_ITERS = max_iters;
      fddp.setup(problem);
      fddp.run(problem, xs_init, us_init);
      const ResultsFDDPTpl<double> &res_fddp = fddp.getResults();
    }
  }
}

int main(int argc, char **argv) {
  shared_ptr<TrajOptProblemTpl<double>> problemptr;
  define_problem(problemptr);

  benchmark::RegisterBenchmark("PROXDDP", &BM_lqr, *problemptr, false);
  benchmark::RegisterBenchmark("FDDP", &BM_lqr, *problemptr, true);

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
}
