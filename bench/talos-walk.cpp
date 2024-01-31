/// @file
/// @brief Benchmark aligator::SolverFDDP against Crocoddyl on a simple example
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA

#include <benchmark/benchmark.h>

#include <crocoddyl/core/solvers/fddp.hpp>
#include <crocoddyl/core/utils/callbacks.hpp>
#include <proxsuite-nlp/fwd.hpp>

#include "talos-walk-utils.hpp"

#include "aligator/core/traj-opt-problem.hpp"
#include "aligator/solvers/fddp/solver-fddp.hpp"
#include "aligator/solvers/proxddp/solver-proxddp.hpp"
#include <proxsuite-nlp/ldlt-allocator.hpp>

using aligator::SolverFDDPTpl;
using aligator::SolverProxDDPTpl;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using TrajOptProblem = aligator::TrajOptProblemTpl<double>;

constexpr double TOL = 1e-4;
constexpr std::size_t maxiters = 100;
constexpr int DEFAULT_NUM_THREADS = 1;

/// Benchmark the full PROXDDP algorithm (aligator::SolverProxDDP)
template <aligator::LQSolverChoice lqsc>
static void BM_aligator(benchmark::State &state) {
  const std::size_t T_ss = (std::size_t)state.range(0);
  const std::size_t T_ds = T_ss / 4;
  const std::size_t nsteps = T_ss * 2 + T_ds * 3;
  const auto num_threads = static_cast<std::size_t>(state.range(1));

  auto problem = defineLocomotionProblem(T_ss, T_ds);

  std::vector<VectorXd> xs_i;
  std::vector<VectorXd> us_i;
  const double mu_init = 1e-8;

  Eigen::VectorXd u0 = Eigen::VectorXd::Zero(22);
  xs_i.assign(nsteps + 1, problem.getInitState());
  us_i.assign(nsteps, u0);

  SolverProxDDPTpl<double> solver(TOL, mu_init, 0., maxiters, aligator::QUIET);

  solver.rollout_type_ = aligator::RolloutType::LINEAR;
  solver.linear_solver_choice = lqsc;
  solver.force_initial_condition_ = true;
  solver.reg_min = 1e-6;
  solver.setNumThreads(num_threads);
  solver.setup(problem);

  for (auto _ : state) {
    bool conv = solver.run(problem, xs_i, us_i);
    if (!conv)
      state.SkipWithError("solver did not converge.");
  }
  state.SetComplexityN(state.range(0));
}

/// Benchmark the FDDP algorithm (aligator::SolverFDDP)
static void BM_FDDP(benchmark::State &state) {
  const std::size_t T_ss = (std::size_t)state.range(0);
  const std::size_t T_ds = T_ss / 4;
  const std::size_t nsteps = T_ss * 2 + T_ds * 3;

  auto problem = defineLocomotionProblem(T_ss, T_ds);

  std::vector<VectorXd> xs_i;
  std::vector<VectorXd> us_i;

  Eigen::VectorXd u0 = Eigen::VectorXd::Zero(22);
  xs_i.assign(nsteps + 1, problem.getInitState());
  us_i.assign(nsteps, u0);

  SolverFDDPTpl<double> solver(TOL, aligator::QUIET);

  solver.force_initial_condition_ = true;
  solver.setup(problem);

  for (auto _ : state) {
    bool conv = solver.run(problem, xs_i, us_i);
    if (!conv)
      state.SkipWithError("solver did not converge.");
  }
  state.SetComplexityN(state.range(0));
}

constexpr auto unit = benchmark::kMillisecond;

static void BaseArgs(benchmark::internal::Benchmark *bench) {
  bench->Complexity()->Unit(unit)->UseRealTime();
}

static void ArgsSerial(benchmark::internal::Benchmark *bench) {
  // bench->ArgName("T_ss")->RangeMultiplier(2)->Range(1 << 3, 3 << 8);
  bench->ArgName("T_ss");
  std::vector<long> T_ss_vec = {60, 80, 100};
  for (auto t : T_ss_vec) {
    bench->Arg(t);
  }
}

static void ArgsParallel(benchmark::internal::Benchmark *bench) {
  bench->ArgNames({"T_ss", "nthreads"});
  std::vector<long> nthreads = {2, 3, 4, 6};
  std::vector<long> T_ss_vec = {60, 80, 100};
  for (auto n : nthreads) {
    for (auto t : T_ss_vec) {
      bench->Args({t, n});
    }
  }
}

int main(int argc, char **argv) {
  benchmark::RegisterBenchmark("FDDP", &BM_FDDP)
      ->Apply(BaseArgs)
      ->Apply(ArgsSerial);
  benchmark::RegisterBenchmark("ALIGATOR_SERIAL",
                               &BM_aligator<aligator::LQSolverChoice::SERIAL>)
      ->Apply(BaseArgs)
      ->Apply(ArgsSerial);
  benchmark::RegisterBenchmark("ALIGATOR_PARALLEL",
                               &BM_aligator<aligator::LQSolverChoice::PARALLEL>)
      ->Apply(BaseArgs)
      ->Apply(ArgsParallel);

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
}
