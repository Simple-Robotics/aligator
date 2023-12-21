
#include "croc-talos-arm.hpp"

#include "aligator/solvers/proxddp/solver-proxddp.hpp"
#include "aligator/solvers/fddp/solver-fddp.hpp"

#include <benchmark/benchmark.h>

using aligator::SolverFDDP;
using aligator::SolverProxDDP;
using Eigen::MatrixXd;
using Eigen::VectorXd;

using T = double;

constexpr T TOL = std::numeric_limits<T>::epsilon();
constexpr std::size_t maxiters = 10;

static void BM_croc_multithread(benchmark::State &state) {
  const std::size_t nsteps = (std::size_t)state.range(0);
  const std::size_t nthreads = (std::size_t)state.range(1);

  std::vector<VectorXd> xs_i, us_i;

  auto croc_problem = defineCrocoddylProblem(nsteps);
  getInitialGuesses(croc_problem, xs_i, us_i);

  (*croc_problem).set_nthreads((int)nthreads);

  croc::SolverFDDP solver(croc_problem);
  solver.set_th_stop(TOL * TOL * static_cast<T>(nsteps));

  for (auto _ : state) {
    solver.solve(xs_i, us_i, maxiters);
  }
}

static void BM_prox_multithread(benchmark::State &state) {
  using aligator::compat::croc::convertCrocoddylProblem;
  const std::size_t nsteps = (std::size_t)state.range(0);
  const std::size_t nthreads = (std::size_t)state.range(1);

  std::vector<VectorXd> xs_i, us_i;

  auto croc_problem = defineCrocoddylProblem(nsteps);
  getInitialGuesses(croc_problem, xs_i, us_i);
  auto problem = convertCrocoddylProblem(croc_problem);
  problem.setNumThreads(nthreads);

  const T mu_init = 1e-6;
  SolverProxDDP<T> solver(TOL, mu_init, 0., maxiters);
  solver.setup(problem);

  for (auto _ : state) {
    solver.run(problem, xs_i, us_i);
  }
}

static void BM_fddp_multithread(benchmark::State &state) {
  using aligator::compat::croc::convertCrocoddylProblem;
  const std::size_t nsteps = (std::size_t)state.range(0);
  const std::size_t nthreads = (std::size_t)state.range(1);

  std::vector<VectorXd> xs_i, us_i;

  auto croc_problem = defineCrocoddylProblem(nsteps);
  getInitialGuesses(croc_problem, xs_i, us_i);
  auto problem = convertCrocoddylProblem(croc_problem);
  problem.setNumThreads(nthreads);

  SolverFDDP<T> solver(TOL);
  solver.max_iters = maxiters;
  solver.setup(problem);

  for (auto _ : state) {
    solver.run(problem, xs_i, us_i);
  }
}

static void custom_args(benchmark::internal::Benchmark *b) {
  std::vector<long> nsteps = {25, 50, 100};
  std::vector<long> nthreads = {1, 2, 4, 6, 8};
  for (auto &ns : nsteps)
    for (auto &nt : nthreads)
      b->Args({ns, nt});
  b->Unit(benchmark::kMillisecond)->UseRealTime();
}

BENCHMARK(BM_prox_multithread)->Apply(custom_args);
BENCHMARK(BM_fddp_multithread)->Apply(custom_args);
BENCHMARK(BM_croc_multithread)->Apply(custom_args);

int main(int argc, char **argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
}
