/// @file
/// @brief Benchmark aligator against Crocoddyl on a nonlinear example
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA

#include <benchmark/benchmark.h>

#include <crocoddyl/core/solvers/fddp.hpp>
#include <crocoddyl/core/utils/callbacks.hpp>

#include "croc-talos-arm.hpp"

#include "aligator/solvers/fddp/solver-fddp.hpp"
#include "aligator/solvers/proxddp/solver-proxddp.hpp"
#include "aligator/compat/crocoddyl/problem-wrap.hpp"

using aligator::SolverFDDPTpl;
using aligator::context::SolverProxDDP;
using Eigen::MatrixXd;
using Eigen::VectorXd;

constexpr double TOL = 1e-16;
constexpr std::size_t maxiters = 10;
constexpr int DEFAULT_NUM_THREADS = 1;

const bool verbose = false;

static void BM_croc_fddp(benchmark::State &state) {
  const std::size_t nsteps = (std::size_t)state.range(0);
  auto croc_problem = defineCrocoddylProblem(nsteps);
#ifdef CROCODDYL_WITH_MULTITHREADING
  croc_problem->set_nthreads((int)DEFAULT_NUM_THREADS);
#endif

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

auto get_verbose_flag(bool verbose) {
  return verbose ? aligator::VERBOSE : aligator::QUIET;
}

static void BM_prox_fddp(benchmark::State &state) {
  const std::size_t nsteps = (std::size_t)state.range(0);
  auto croc_problem = defineCrocoddylProblem(nsteps);
  auto prob_wrap =
      aligator::compat::croc::convertCrocoddylProblem(croc_problem);

  std::vector<VectorXd> xs_i;
  std::vector<VectorXd> us_i;
  getInitialGuesses(croc_problem, xs_i, us_i);

  SolverFDDPTpl<double> solver(TOL, get_verbose_flag(verbose));
  solver.max_iters = maxiters;
  solver.setNumThreads(DEFAULT_NUM_THREADS);
  solver.setup(prob_wrap);

  for (auto _ : state) {
    solver.run(prob_wrap, xs_i, us_i);
  }
  state.SetComplexityN(state.range(0));
}

/// Benchmark the full PROXDDP algorithm (aligator::SolverProxDDP)
template <uint NPROC> void BM_aligator(benchmark::State &state) {
  const std::size_t nsteps = (std::size_t)state.range(0);
  auto croc_problem = defineCrocoddylProblem(nsteps);
  auto prob_wrap =
      aligator::compat::croc::convertCrocoddylProblem(croc_problem);

  std::vector<VectorXd> xs_i;
  std::vector<VectorXd> us_i;
  getInitialGuesses(croc_problem, xs_i, us_i);

  const double mu_init = 1e-10;
  SolverProxDDP solver(TOL, mu_init, maxiters);
  solver.verbose_ = get_verbose_flag(verbose);
  solver.setNumThreads(NPROC);
  solver.setup(prob_wrap);

  for (auto _ : state) {
    solver.run(prob_wrap, xs_i, us_i);
  }
  state.SetComplexityN(state.range(0));
}

constexpr long nmin = 50;
constexpr long nmax = 450;
constexpr long ns = 50;
constexpr auto unit = benchmark::kMillisecond;

void CustomArgs(benchmark::internal::Benchmark *bench) {
  bench->Arg(5)
      ->Arg(20)
      ->DenseRange(nmin, nmax, ns)
      ->Unit(unit)
      ->Complexity()
      ->UseRealTime();
};

BENCHMARK(BM_croc_fddp)->Apply(CustomArgs);
BENCHMARK(BM_prox_fddp)->Apply(CustomArgs);
BENCHMARK_TEMPLATE(BM_aligator, 2)->Apply(CustomArgs);
BENCHMARK_TEMPLATE(BM_aligator, 4)->Apply(CustomArgs);
BENCHMARK_TEMPLATE(BM_aligator, 6)->Apply(CustomArgs);
BENCHMARK_TEMPLATE(BM_aligator, 8)->Apply(CustomArgs);

int main(int argc, char **argv) {

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
}
