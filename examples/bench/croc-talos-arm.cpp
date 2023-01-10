/// @file
/// @brief Benchmark proxddp::SolverFDDP against Crocoddyl on a simple example
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA

#include "croc-talos-arm.hpp"

#include "proxddp/fddp/solver-fddp.hpp"
#include "proxddp/core/solver-proxddp.hpp"

#include <benchmark/benchmark.h>

constexpr double TOL = 1e-16;
constexpr std::size_t maxiters = 10;

bool verbose = true;

using Eigen::MatrixXd;
using Eigen::VectorXd;

void getInitialGuesses(
    const boost::shared_ptr<croc::ShootingProblem> &croc_problem,
    std::vector<VectorXd> &xs_i, std::vector<VectorXd> &us_i) {

  const std::size_t nsteps = croc_problem->get_T();
  const auto &x0 = croc_problem->get_x0();
  const long nu = (long)croc_problem->get_nu_max();
  VectorXd u0 = VectorXd::Zero(nu);

  xs_i.assign(nsteps + 1, x0);
  us_i.assign(nsteps, u0);
}

static void BM_croc_fddp(benchmark::State &state) {
  const std::size_t nsteps = (std::size_t)state.range(0);
  auto croc_problem = defineCrocoddylProblem(nsteps);

  std::vector<VectorXd> xs_i;
  std::vector<VectorXd> us_i;
  getInitialGuesses(croc_problem, xs_i, us_i);

  croc::SolverFDDP solver(croc_problem);
  const double croc_tol = TOL * TOL * (double)nsteps;
  solver.set_th_stop(croc_tol);
  if (verbose)
    solver.setCallbacks({boost::make_shared<croc::CallbackVerbose>()});

  for (auto _ : state) {
    solver.solve(xs_i, us_i, maxiters);
  }
}

static void BM_prox_fddp(benchmark::State &state) {
  const std::size_t nsteps = (std::size_t)state.range(0);
  using proxddp::VerboseLevel;
  auto croc_problem = defineCrocoddylProblem(nsteps);
  auto prob_wrap = proxddp::compat::croc::convertCrocoddylProblem(croc_problem);

  std::vector<VectorXd> xs_i;
  std::vector<VectorXd> us_i;
  getInitialGuesses(croc_problem, xs_i, us_i);

  auto verbose = VerboseLevel::QUIET;
  proxddp::SolverFDDP<double> solver(TOL, verbose);
  solver.max_iters = maxiters;
  solver.setup(prob_wrap);

  for (auto _ : state) {
    solver.run(prob_wrap, xs_i, us_i);
  }
  state.SetComplexityN(state.range(0));
}

/// Benchmark the full PROXDDP algorithm (proxddp::SolverProxDDP)
static void BM_proxddp(benchmark::State &state) {
  const std::size_t nsteps = (std::size_t)state.range(0);
  using proxddp::LDLTChoice;
  using proxddp::SolverProxDDP;
  using proxddp::VerboseLevel;
  auto croc_problem = defineCrocoddylProblem(nsteps);
  auto prob_wrap = proxddp::compat::croc::convertCrocoddylProblem(croc_problem);

  std::vector<VectorXd> xs_i;
  std::vector<VectorXd> us_i;
  getInitialGuesses(croc_problem, xs_i, us_i);

  const double mu0 = 1e-4;
  SolverProxDDP<double> solver(TOL, mu0, 0., maxiters,
                               verbose ? VerboseLevel::VERBOSE
                                       : VerboseLevel::QUIET);
  solver.ldlt_algo_choice_ = LDLTChoice::DENSE;
  solver.max_refinement_steps_ = 0;
  solver.setup(prob_wrap);

  for (auto _ : state) {
    solver.run(prob_wrap, xs_i, us_i);
  }
  state.SetComplexityN(state.range(0));
}

int main(int argc, char **argv) {

  constexpr long nmin = 50;
  constexpr long nmax = 450;
  constexpr long ns = 50;
  auto unit = benchmark::kMillisecond;
  auto registerWithOpts = [&](auto name, auto fn) {
    benchmark::RegisterBenchmark(name, fn)
        ->DenseRange(nmin, nmax, ns)
        ->Unit(unit)
        ->Complexity();
  };
  registerWithOpts("croc::FDDP", &BM_croc_fddp);
  registerWithOpts("proxddp::FDDP", &BM_prox_fddp);
  registerWithOpts("proxddp::PROXDDP", &BM_proxddp);

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
}
