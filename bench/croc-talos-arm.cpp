/// @file
/// @brief Benchmark aligator::SolverFDDP against Crocoddyl on a simple example
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA

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
using proxsuite::nlp::LDLTChoice;

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
#ifdef ALIGATOR_MULTITHREADING
  prob_wrap.setNumThreads(DEFAULT_NUM_THREADS);
#endif

  std::vector<VectorXd> xs_i;
  std::vector<VectorXd> us_i;
  getInitialGuesses(croc_problem, xs_i, us_i);

  SolverFDDPTpl<double> solver(TOL, get_verbose_flag(verbose));
  solver.max_iters = maxiters;
  solver.setup(prob_wrap);

  for (auto _ : state) {
    solver.run(prob_wrap, xs_i, us_i);
  }
  state.SetComplexityN(state.range(0));
}

/// Benchmark the full PROXDDP algorithm (aligator::SolverProxDDP)
template <LDLTChoice choice> static void BM_aligator(benchmark::State &state) {
  const std::size_t nsteps = (std::size_t)state.range(0);
  using aligator::LDLTChoice;
  auto croc_problem = defineCrocoddylProblem(nsteps);
  auto prob_wrap =
      aligator::compat::croc::convertCrocoddylProblem(croc_problem);
#ifdef ALIGATOR_MULTITHREADING
  prob_wrap.setNumThreads(DEFAULT_NUM_THREADS);
#endif

  std::vector<VectorXd> xs_i;
  std::vector<VectorXd> us_i;
  getInitialGuesses(croc_problem, xs_i, us_i);

  const double mu0 = 1e-4;
  SolverProxDDP solver(TOL, mu0, 0., maxiters, get_verbose_flag(verbose));
  solver.maxRefinementSteps_ = 0;
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
        ->Arg(5)
        ->Arg(20)
        ->DenseRange(nmin, nmax, ns)
        ->Unit(unit)
        ->Complexity()
        ->UseRealTime();
  };
  registerWithOpts("croc::FDDP", &BM_croc_fddp);
  registerWithOpts("aligator::FDDP", &BM_prox_fddp);
  registerWithOpts("aligator::ALIGATOR_DENSE", &BM_aligator<LDLTChoice::DENSE>);
  registerWithOpts("aligator::ALIGATOR_BLOCK",
                   &BM_aligator<LDLTChoice::BLOCKSPARSE>);
  registerWithOpts("aligator::ALIGATOR_BUNCHKAUFMAN",
                   &BM_aligator<LDLTChoice::BUNCHKAUFMAN>);
  registerWithOpts("aligator::ALIGATOR_EIGLDL",
                   &BM_aligator<LDLTChoice::EIGEN>);
#ifdef PROXSUITE_NLP_USE_PROXSUITE_LDLT
  registerWithOpts("aligator::ALIGATOR_PSUITE",
                   &BM_aligator<LDLTChoice::PROXSUITE>);
#endif

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
}
