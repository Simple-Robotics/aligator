/// @file
/// @brief Bench simple Riccati impls
/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA

#include <benchmark/benchmark.h>

#include "aligator/gar/riccati-kernel.hpp"
#include "aligator/threads.hpp"

#include "../tests/gar/test_util.hpp"

using namespace aligator::gar;

static void BM_riccati_impl(benchmark::State &state) {
  const uint horz = (uint)state.range(0);
  const uint nx = (uint)state.range(1);
  const uint nu = (uint)state.range(2);

  VectorXs x0 = VectorXs::NullaryExpr(nx, normal_unary_op{});
  auto problem = generate_problem(x0, horz, nx, nu);
  using Kernel = ProximalRiccatiKernel<double>;

  for (auto _ : state) {
  }
}

BENCHMARK(BM_riccati_impl);

int main(int argc, char **argv) {
  aligator::omp::set_default_options(aligator::omp::get_available_threads());

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
}
