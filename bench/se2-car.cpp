#include <benchmark/benchmark.h>

#include "se2-car.hpp"

#include "aligator/solvers/proxddp/solver-proxddp.hpp"

static void bench_serial(benchmark::State &state) {
  std::size_t nsteps = static_cast<std::size_t>(state.range(0));
  const auto prob = create_se2_problem(nsteps);
  const T mu_init = 1e-3;
  SolverProxDDPTpl<T> solver(1e-3, mu_init);
  solver.max_iters = 4;
  solver.setup(prob);

  for (auto _ : state) {
    solver.run(prob);
  }
}

static void bench_parallel(benchmark::State &state) {
  std::size_t nsteps = static_cast<std::size_t>(state.range(0));
  std::size_t num_threads = static_cast<std::size_t>(state.range(1));
  const auto prob = create_se2_problem(nsteps);
  const T mu_init = 1e-3;
  SolverProxDDPTpl<T> solver(1e-3, mu_init);
  solver.max_iters = 4;
  solver.rollout_type_ = RolloutType::LINEAR;
  solver.linear_solver_choice = LQSolverChoice::PARALLEL;
  solver.setNumThreads(num_threads);
  solver.setup(prob);

  for (auto _ : state) {
    solver.run(prob);
  }
}

constexpr auto timeUnit = benchmark::kMillisecond;

static void CustomArgs(benchmark::internal::Benchmark *bench) {
  bench->Unit(timeUnit)->UseRealTime();
  bench->ArgNames({"nsteps"});
  for (long e = 2; e <= 10; e++) {
    bench->Args({20 * e});
  }
}

static void ParallelArgs(benchmark::internal::Benchmark *bench) {
  bench->Unit(timeUnit)->UseRealTime();
  bench->ArgNames({"nsteps", "nthreads"});
  for (long e = 2; e <= 10; e++) {
    for (long nt = 1; nt <= 3; nt++)
      bench->Args({20 * e, 2 * nt});
  }
}

BENCHMARK(bench_serial)->Apply(CustomArgs);
#ifdef ALIGATOR_MULTITHREADING
BENCHMARK(bench_parallel)->Apply(ParallelArgs);
#endif

BENCHMARK_MAIN();
