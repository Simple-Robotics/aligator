/// @file
/// @brief Benchmark solvers from GAR
/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA

#include <benchmark/benchmark.h>

#include "aligator/gar/proximal-riccati.hpp"
#include "aligator/gar/parallel-solver.hpp"
#include "aligator/gar/dense-riccati.hpp"
#include "aligator/gar/utils.hpp"

#include "aligator/threads.hpp"

#include "../tests/gar/test_util.hpp"

using namespace aligator::gar;

static constexpr uint nx = 36;
static constexpr uint nu = 12;
static constexpr double mueq = 1e-11;

static void BM_serial(benchmark::State &state) {
  uint horz = (uint)state.range(0);
  VectorXs x0 = VectorXs::NullaryExpr(nx, normal_unary_op{});
  const LqrProblemTpl<double> problem = generateLqProblem(x0, horz, nx, nu);
  ProximalRiccatiSolver<double> solver(problem);
  auto [xs, us, vs, lbdas] = lqrInitializeSolution(problem);
  for (auto _ : state) {
    solver.backward(mueq);
    solver.forward(xs, us, vs, lbdas);
  }
}

#ifdef ALIGATOR_MULTITHREADING
template <uint NPROC> static void BM_parallel(benchmark::State &state) {
  uint horz = (uint)state.range(0);
  VectorXs x0 = VectorXs::NullaryExpr(nx, normal_unary_op{});
  LqrProblemTpl<double> problem = generateLqProblem(x0, horz, nx, nu);
  ParallelRiccatiSolver<double> solver(problem, NPROC);
  auto [xs, us, vs, lbdas] = lqrInitializeSolution(problem);
  for (auto _ : state) {
    solver.backward(mueq);
    solver.forward(xs, us, vs, lbdas);
  }
}
#endif

static void BM_stagedense(benchmark::State &state) {
  uint horz = (uint)state.range(0);
  VectorXs x0 = VectorXs::NullaryExpr(nx, normal_unary_op{});
  LqrProblemTpl<double> problem = generateLqProblem(x0, horz, nx, nu);
  RiccatiSolverDense<double> solver(problem);
  auto [xs, us, vs, lbdas] = lqrInitializeSolution(problem);
  for (auto _ : state) {
    solver.backward(mueq);
    solver.forward(xs, us, vs, lbdas);
  }
}

static void customArgs(benchmark::internal::Benchmark *b) {
  for (uint e = 4; e <= 10; e++) {
    b->Arg(1 << e);
  }
  b->Unit(benchmark::kMillisecond);
  b->UseRealTime();
}

BENCHMARK(BM_serial)->Apply(customArgs);
BENCHMARK(BM_stagedense)->Apply(customArgs);
#ifdef ALIGATOR_MULTITHREADING
BENCHMARK_TEMPLATE(BM_parallel, 2)->Apply(customArgs);
BENCHMARK_TEMPLATE(BM_parallel, 3)->Apply(customArgs);
BENCHMARK_TEMPLATE(BM_parallel, 4)->Apply(customArgs);
BENCHMARK_TEMPLATE(BM_parallel, 6)->Apply(customArgs);
#endif

int main(int argc, char **argv) {
  aligator::omp::set_default_options(aligator::omp::get_available_threads());

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
}
