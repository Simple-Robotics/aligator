/// @file
/// @brief Benchmark solvers from GAR
/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA

#include <benchmark/benchmark.h>

#include "aligator/gar/riccati.hpp"
#include "aligator/gar/parallel-solver.hpp"
#include "aligator/gar/dense-riccati.hpp"

#include "aligator/gar/utils.hpp"

#include "../tests/gar/test_util.hpp"

using namespace aligator::gar;

const uint nx = 36;
const uint nu = 12;

static void BM_serial(benchmark::State &state) {
  uint horz = (uint)state.range(0);
  VectorXs x0 = VectorXs::NullaryExpr(nx, normal_unary_op{});
  const LQRProblemTpl<double> problem = generate_problem(x0, horz, nx, nu);
  ProximalRiccatiSolver<double> solver(problem);
  const double mu = 1e-11;
  auto [xs, us, vs, lbdas] = lqrInitializeSolution(problem);
  for (auto _ : state) {
    solver.backward(mu, mu);
    solver.forward(xs, us, vs, lbdas);
  }
}

template <uint NPROC> static void BM_parallel(benchmark::State &state) {
  uint horz = (uint)state.range(0);
  VectorXs x0 = VectorXs::NullaryExpr(nx, normal_unary_op{});
  LQRProblemTpl<double> problem = generate_problem(x0, horz, nx, nu);
  ParallelRiccatiSolver<double> solver(problem, NPROC);
  const double mu = 1e-11;
  auto [xs, us, vs, lbdas] = lqrInitializeSolution(problem);
  for (auto _ : state) {
    solver.backward(mu, mu);
    solver.forward(xs, us, vs, lbdas);
  }
}

static void BM_stagedense(benchmark::State &state) {
  uint horz = (uint)state.range(0);
  VectorXs x0 = VectorXs::NullaryExpr(nx, normal_unary_op{});
  LQRProblemTpl<double> problem = generate_problem(x0, horz, nx, nu);
  RiccatiSolverDense<double> solver(problem);
  const double mu = 1e-11;
  auto [xs, us, vs, lbdas] = lqrInitializeSolution(problem);
  for (auto _ : state) {
    solver.backward(mu, mu);
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
BENCHMARK_TEMPLATE(BM_parallel, 2)->Apply(customArgs);
BENCHMARK_TEMPLATE(BM_parallel, 3)->Apply(customArgs);
BENCHMARK_TEMPLATE(BM_parallel, 4)->Apply(customArgs);
BENCHMARK_TEMPLATE(BM_parallel, 5)->Apply(customArgs);
BENCHMARK_TEMPLATE(BM_parallel, 6)->Apply(customArgs);

int main(int argc, char **argv) {
  aligator::omp::set_default_options(aligator::omp::get_available_threads());

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
}
