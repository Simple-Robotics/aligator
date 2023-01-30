/// @file
/// @brief Benchmark proxddp::SolverFDDP against Crocoddyl on a simple example
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA

#include "croc-talos-arm.hpp"

#include "proxddp/fddp/solver-fddp.hpp"
#include "proxddp/core/solver-proxddp.hpp"

#include <benchmark/benchmark.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using proxddp::LDLTChoice;
using proxddp::SolverFDDP;
using proxddp::SolverProxDDP;
using proxddp::VerboseLevel;

constexpr double TOL = 1e-16;
constexpr std::size_t maxiters = 10;

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision,
                                       Eigen::DontAlignCols, ", ", "\n");

const bool verbose = false;

struct extract_kkt_matrix_callback : proxddp::helpers::base_callback<double> {
  std::string filepath;
  extract_kkt_matrix_callback(std::string const &filepath)
      : filepath(filepath) {}
  void call(const Workspace &ws_, const Results &) {
    const auto &ws = static_cast<const proxddp::context::Workspace &>(ws_);
    std::ofstream file(filepath);
    for (std::size_t t = 0; t < ws.kkt_mats_.size(); t++) {
      MatrixXd const &w = ws.kkt_mats_[t];

      file << w.format(CSVFormat);
      file << "\n\n";
    }
  }
};

void getInitialGuesses(
    const boost::shared_ptr<croc::ShootingProblem> &croc_problem,
    std::vector<VectorXd> &xs_i, std::vector<VectorXd> &us_i) {

  const std::size_t nsteps = croc_problem->get_T();
  const auto &x0 = croc_problem->get_x0();
  const long nu = (long)croc_problem->get_nu_max();
  VectorXd u0 = VectorXd::Zero(nu);

  xs_i.assign(nsteps + 1, x0);
  us_i.assign(nsteps, u0);
}

static void BM_croc_fddp(benchmark::State &state) {
  const std::size_t nsteps = (std::size_t)state.range(0);
  auto croc_problem = defineCrocoddylProblem(nsteps);

  std::vector<VectorXd> xs_i;
  std::vector<VectorXd> us_i;
  getInitialGuesses(croc_problem, xs_i, us_i);

  croc::SolverFDDP solver(croc_problem);
  const double croc_tol = TOL * TOL * (double)nsteps;
  solver.set_th_stop(croc_tol);
  if (verbose)
    solver.setCallbacks({boost::make_shared<croc::CallbackVerbose>()});

  for (auto _ : state) {
    solver.solve(xs_i, us_i, maxiters);
  }
}

auto get_verbose_flag(bool verbose) {
  return verbose ? VerboseLevel::VERBOSE : VerboseLevel::QUIET;
}

static void BM_prox_fddp(benchmark::State &state) {
  const std::size_t nsteps = (std::size_t)state.range(0);
  auto croc_problem = defineCrocoddylProblem(nsteps);
  auto prob_wrap = proxddp::compat::croc::convertCrocoddylProblem(croc_problem);

  std::vector<VectorXd> xs_i;
  std::vector<VectorXd> us_i;
  getInitialGuesses(croc_problem, xs_i, us_i);

  SolverFDDP<double> solver(TOL, get_verbose_flag(verbose));
  solver.max_iters = maxiters;
  solver.setup(prob_wrap);

  for (auto _ : state) {
    solver.run(prob_wrap, xs_i, us_i);
  }
  state.SetComplexityN(state.range(0));
}

/// Benchmark the full PROXDDP algorithm (proxddp::SolverProxDDP)
template <LDLTChoice choice> static void BM_proxddp(benchmark::State &state) {
  const std::size_t nsteps = (std::size_t)state.range(0);
  using proxddp::LDLTChoice;
  auto croc_problem = defineCrocoddylProblem(nsteps);
  auto prob_wrap = proxddp::compat::croc::convertCrocoddylProblem(croc_problem);

  std::vector<VectorXd> xs_i;
  std::vector<VectorXd> us_i;
  getInitialGuesses(croc_problem, xs_i, us_i);

  const double mu0 = 1e-4;
  SolverProxDDP<double> solver(TOL, mu0, 0., maxiters,
                               get_verbose_flag(verbose));
  solver.ldlt_algo_choice_ = choice;
  solver.max_refinement_steps_ = 0;
  solver.setup(prob_wrap);

  // solver.registerCallback(
  //     std::make_shared<extract_kkt_matrix_callback>("stupid_eigen_files.csv"));

  for (auto _ : state) {
    solver.run(prob_wrap, xs_i, us_i);
  }
  state.SetComplexityN(state.range(0));
}

int main(int argc, char **argv) {

  constexpr long nmin = 50;
  constexpr long nmax = 450;
  constexpr long ns = 50;
  auto unit = benchmark::kMillisecond;
  auto registerWithOpts = [&](auto name, auto fn) {
    benchmark::RegisterBenchmark(name, fn)
        ->Arg(5)
        ->Arg(20)
        ->DenseRange(nmin, nmax, ns)
        ->Unit(unit)
        ->ArgName("nsteps")
        ->Complexity()
        ->UseRealTime();
  };
  registerWithOpts("croc::FDDP", &BM_croc_fddp);
  registerWithOpts("proxddp::FDDP", &BM_prox_fddp);
  registerWithOpts("proxddp::PROXDDP_DENSE", &BM_proxddp<LDLTChoice::DENSE>);
  registerWithOpts("proxddp::PROXDDP_BLOCK", &BM_proxddp<LDLTChoice::BLOCKED>);
  registerWithOpts("proxddp::PROXDDP_EIGLDL", &BM_proxddp<LDLTChoice::EIGEN>);

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
}
