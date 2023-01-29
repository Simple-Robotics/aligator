/// @file
/// @brief Linear-quadratic regulator

#include "proxddp/core/solver-proxddp.hpp"
#include "proxddp/fddp/solver-fddp.hpp"
#include "proxddp/utils/rollout.hpp"
#include "proxddp/modelling/quad-costs.hpp"

#include "proxddp/modelling/linear-discrete-dynamics.hpp"

#include <benchmark/benchmark.h>

using namespace proxddp;

using T = double;
constexpr T TOL = 1e-7;
auto verbose = VerboseLevel::QUIET;
using StageModel = StageModelTpl<T>;
using TrajOptProblem = TrajOptProblemTpl<T>;
using Eigen::MatrixXd;
using Eigen::VectorXd;

const std::size_t max_iters = 2;

TrajOptProblem define_problem(const std::size_t nsteps) {
  const int dim = 2;
  const int nu = 2;
  MatrixXd A(dim, dim);
  MatrixXd B(dim, nu);
  VectorXd c_(dim);
  A.setIdentity();
  B << -0.6, 0.3, 0., 1.;
  c_ << 0.1, 0.;

  MatrixXd w_x(dim, dim), w_u(nu, nu);
  w_x << 2., 0., 0., 1.;
  w_u.setIdentity();
  w_u *= 1e-2;

  using dynamics::LinearDiscreteDynamicsTpl;
  auto dynptr = std::make_shared<LinearDiscreteDynamicsTpl<T>>(A, B, c_);
  auto spaceptr = dynptr->space_next_;

  auto rcost = std::make_shared<QuadraticCostTpl<T>>(w_x, w_u);
  auto stage = std::make_shared<StageModel>(rcost, dynptr);

  VectorXd x0(dim);
  x0 << 1., -0.1;

  auto &term_cost = rcost;
  TrajOptProblem problem(x0, nu, spaceptr, term_cost);
  for (std::size_t i = 0; i < nsteps; i++) {
    problem.addStage(stage);
  }
  return problem;
}

#define SETUP_PROBLEM_VARS(state)                                              \
  auto problem = define_problem((std::size_t)state.range(0));                  \
  const auto &dynamics = problem.stages_[0]->dyn_model();                      \
  const VectorXd &x0 = problem.getInitState();                                 \
  std::vector<VectorXd> us_init;                                               \
  us_default_init(problem, us_init);                                           \
  std::vector<VectorXd> xs_init = rollout(dynamics, x0, us_init)

static void BM_lqr_prox(benchmark::State &state) {
  SETUP_PROBLEM_VARS(state);
  const T mu_init = 1e-6;
  const T rho_init = 0.;
  SolverProxDDP<T> solver(TOL, mu_init, rho_init, max_iters, verbose);
  solver.ldlt_algo_choice_ = static_cast<LDLTChoice>(state.range(1));
  solver.max_refinement_steps_ = 0;
  solver.setup(problem);

  for (auto _ : state) {
    bool conv = solver.run(problem, xs_init, us_init);
    if (!conv)
      state.SkipWithError("solver did not converge.");
  }
  state.SetComplexityN(state.range(0));
}

static void BM_lqr_fddp(benchmark::State &state) {
  SETUP_PROBLEM_VARS(state);
  SolverFDDP<T> fddp(TOL, verbose);
  fddp.max_iters = max_iters;
  fddp.setup(problem);

  for (auto _ : state) {
    bool conv = fddp.run(problem, xs_init, us_init);
    if (!conv)
      state.SkipWithError("solver did not converge.");
  }
  state.SetComplexityN(state.range(0));
}

int main(int argc, char **argv) {

  auto registerOpts = [&](auto name, auto fn) {
    return benchmark::RegisterBenchmark(name, fn)
        ->ArgNames({"nsteps", "ldlt_choice"})
        ->Complexity()
        ->Unit(benchmark::kMicrosecond);
  };

  registerOpts("PROXDDP", &BM_lqr_prox)->Apply([](auto p) {
    for (LDLTChoice ch :
         {LDLTChoice::DENSE, LDLTChoice::EIGEN, LDLTChoice::BLOCKED}) {
      for (std::size_t i = 3; i < 9; ++i) {
        p = p->Args({1 << i, (long)ch});
      }
    }
  });

  registerOpts("FDDP", &BM_lqr_fddp)->RangeMultiplier(2)->Range(1 << 3, 1 << 9);

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
}
