/// @file
/// @brief Benchmark solvers from GAR
/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA

#include <benchmark/benchmark.h>

#include "aligator/core/mimalloc-resource.hpp"
#include "aligator/gar/proximal-riccati.hpp"
#include "aligator/gar/parallel-solver.hpp"
#include "aligator/gar/dense-riccati.hpp"
#include "aligator/gar/utils.hpp"

#include "aligator/threads.hpp"

#include "../tests/gar/test_util.hpp"

using namespace aligator::gar;

static constexpr uint nx = 36;
static constexpr uint nu = 12;
static constexpr uint nc = 32;
static constexpr double mueq = 1e-11;
static std::mt19937 rng;
static normal_unary_op normal_op{rng};

static aligator::mimalloc_resource mim_resource;

auto get_allocator(int64_t ID) -> aligator::polymorphic_allocator {
  static std::pmr::memory_resource *RESOURCES[2] = {
      std::pmr::get_default_resource(), &mim_resource};
  assert(ID < 2);
  return aligator::polymorphic_allocator{RESOURCES[ID]};
}

#define GET_PROBLEM(state)                                                     \
  auto allocator = get_allocator(state.range(1));                              \
  uint horz = (uint)state.range(0);                                            \
  VectorXs x0 = VectorXs::NullaryExpr(nx, normal_op);                          \
  LqrProblemTpl<double> problem =                                              \
      generateLqProblem(rng, x0, horz, nx, nu, 0, nc, true, allocator)

static void BM_serial(benchmark::State &state) {
  GET_PROBLEM(state);
  ProximalRiccatiSolver<double> solver(problem);
  auto [xs, us, vs, lbdas] = lqrInitializeSolution(problem);
  for (auto _ : state) {
    solver.backward(mueq);
    solver.forward(xs, us, vs, lbdas);
  }
}

#ifdef ALIGATOR_MULTITHREADING
template <uint NPROC> static void BM_parallel(benchmark::State &state) {
  GET_PROBLEM(state);
  ParallelRiccatiSolver<double> solver(problem, NPROC);
  auto [xs, us, vs, lbdas] = lqrInitializeSolution(problem);
  for (auto _ : state) {
    solver.backward(mueq);
    solver.forward(xs, us, vs, lbdas);
  }
}
#endif

static void BM_stagedense(benchmark::State &state) {
  GET_PROBLEM(state);
  RiccatiSolverDense<double> solver(problem);
  auto [xs, us, vs, lbdas] = lqrInitializeSolution(problem);
  for (auto _ : state) {
    solver.backward(mueq);
    solver.forward(xs, us, vs, lbdas);
  }
}

static void customArgs(benchmark::internal::Benchmark *b) {
  for (uint e = 4; e <= 10; e++) {
    b->Args({1 << e, 0});
    b->Args({1 << e, 1});
  }
  b->ArgNames({"N", "alloc"});
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
