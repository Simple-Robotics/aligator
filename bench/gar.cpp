/// @file
/// @brief Benchmark solvers from GAR

#include <benchmark/benchmark.h>

#include "aligator/gar/riccati.hpp"
#include "aligator/gar/parallel-solver.hpp"

#include "../tests/gar/test_util.hpp"

using namespace aligator::gar;

using LQProblem = LQRProblemTpl<double>;

const uint nx = 8;
const uint nu = 8;

static void BM_serial(benchmark::State &state) {
  uint horz = (uint)state.range(0);
  VectorXs x0 = VectorXs::NullaryExpr(nx, normal_unary_op{});
  const LQProblem problem = generate_problem(x0, horz, nx, nu);
  ProximalRiccatiSolver<double> solver(problem);
  const double mu = 1e-11;
  auto [xs, us, vs, lbdas] = lqrInitializeSolution(problem);
  for (auto _ : state) {
    solver.backward(mu, mu);
    solver.forward(xs, us, vs, lbdas);
  }
}

static void BM_parallel(benchmark::State &state) {
  uint horz = (uint)state.range(0);
  VectorXs x0 = VectorXs::NullaryExpr(nx, normal_unary_op{});
  const LQProblem problem = generate_problem(x0, horz, nx, nu);
  ParallelRiccatiSolver<double> solver(problem);
  const double mu = 1e-11;
  auto [xs, us, vs, lbdas] = lqrInitializeSolution(problem);
  for (auto _ : state) {
    solver.backward(mu, mu);
    solver.forward(xs, us, vs, lbdas);
  }
}

static void customArgs(benchmark::internal::Benchmark *b) {
  for (uint e = 3; e <= 7; e++) {
    b->Arg(1 << e);
  }
  b->Unit(benchmark::kMillisecond);
  b->UseRealTime();
}

BENCHMARK(BM_serial)->Apply(customArgs);
BENCHMARK(BM_parallel)->Apply(customArgs);

BENCHMARK_MAIN();
