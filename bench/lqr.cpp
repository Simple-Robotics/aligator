/// @file
/// @brief Linear-quadratic regulator

#include "aligator/solvers/proxddp/solver-proxddp.hpp"
#include "aligator/solvers/fddp/solver-fddp.hpp"
#include "aligator/utils/rollout.hpp"
#include "aligator/modelling/costs/quad-costs.hpp"

#include "aligator/modelling/linear-discrete-dynamics.hpp"

#include <benchmark/benchmark.h>

using namespace aligator;

using T = double;
constexpr T TOL = 1e-7;
auto verbose = VerboseLevel::QUIET;
using StageModel = StageModelTpl<T>;
using TrajOptProblem = TrajOptProblemTpl<T>;
using Eigen::MatrixXd;
using Eigen::VectorXd;

const std::size_t max_iters = 2;

TrajOptProblem define_problem(const std::size_t nsteps, const int dim = 56,
                              const int nu = 22) {
  MatrixXd A(dim, dim);
  MatrixXd B(dim, nu);
  VectorXd c_(dim);
  A.setIdentity();
  B.setIdentity();
  c_.setConstant(0.1);

  MatrixXd w_x(dim, dim), w_u(nu, nu);
  w_x.setIdentity();
  w_x(0, 0) = 2.;
  w_u.setIdentity();
  w_u *= 1e-2;

  using Dynamics = dynamics::LinearDiscreteDynamicsTpl<T>;
  using QuadCost = QuadraticCostTpl<T>;
  auto dynptr = Dynamics(A, B, c_);
  auto space = dynptr.space_next_;

  auto rcost = std::make_shared<QuadCost>(w_x, w_u);
  auto stage = std::make_shared<StageModel>(rcost, dynptr);
  auto term_cost = rcost;

  VectorXd x0(dim);
  x0.setRandom();

  TrajOptProblem problem(x0, nu, space, term_cost);
  for (std::size_t i = 0; i < nsteps; i++) {
    problem.addStage(stage);
  }
  return problem;
}

#define SETUP_PROBLEM_VARS(nsteps)                                             \
  auto problem = define_problem(nsteps);                                       \
  const auto &dynamics = *problem.stages_[0]->dynamics_;                       \
  const VectorXd &x0 = problem.getInitState();                                 \
  std::vector<VectorXd> us_init;                                               \
  us_default_init(problem, us_init);                                           \
  std::vector<VectorXd> xs_init = rollout(dynamics, x0, us_init)

template <LQSolverChoice lqsc>
static void BM_lqr_prox(benchmark::State &state) {
  const auto nsteps = static_cast<std::size_t>(state.range(0));
  SETUP_PROBLEM_VARS(nsteps);
  const T mu_init = 1e-10;
  const auto num_threads = static_cast<std::size_t>(state.range(1));
  SolverProxDDPTpl<T> solver(TOL, mu_init, 0., max_iters, verbose);
  solver.linear_solver_choice = lqsc;
  solver.rollout_type_ = RolloutType::LINEAR;
  solver.force_initial_condition_ = false;
  solver.setNumThreads(num_threads);
  solver.setup(problem);

  for (auto _ : state) {
    bool conv = solver.run(problem, xs_init, us_init);
    if (!conv)
      state.SkipWithError("solver did not converge.");
  }
  state.SetComplexityN(state.range(0));
}

static void BM_lqr_fddp(benchmark::State &state) {
  const auto nsteps = static_cast<std::size_t>(state.range(0));
  SETUP_PROBLEM_VARS(nsteps);
  SolverFDDPTpl<T> fddp(TOL, verbose);
  fddp.max_iters = max_iters;
  fddp.setup(problem);

  for (auto _ : state) {
    bool conv = fddp.run(problem, xs_init, us_init);
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
  bench->ArgName("nsteps")->RangeMultiplier(2)->Range(1 << 3, 1 << 9);
}

static void ArgsParallel(benchmark::internal::Benchmark *bench) {
  bench->ArgNames({"nsteps", "nthreads"});
  std::vector<long> nthreads = {2, 3, 4, 6};
  for (auto n : nthreads) {
    for (size_t j = 4; j < 9; j++) {
      bench->Args({1 << j, n});
    }
  }
}

int main(int argc, char **argv) {

  benchmark::RegisterBenchmark("FDDP", &BM_lqr_fddp)
      ->Apply(BaseArgs)
      ->Apply(ArgsSerial);
  benchmark::RegisterBenchmark("ALIGATOR_SERIAL",
                               &BM_lqr_prox<LQSolverChoice::SERIAL>)
      ->Apply(BaseArgs)
      ->Apply(ArgsSerial);
  benchmark::RegisterBenchmark("ALIGATOR_PARALLEL",
                               &BM_lqr_prox<LQSolverChoice::PARALLEL>)
      ->Apply(BaseArgs)
      ->Apply(ArgsParallel);

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
}
