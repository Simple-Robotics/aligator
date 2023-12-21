#include "croc-talos-arm.hpp"

#include "aligator/solvers/proxddp/solver-proxddp.hpp"
#include <proxsuite-nlp/tests/cnpy.hpp>
#include <fmt/ostream.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

using aligator::SolverProxDDP;

constexpr double TOL = 1e-4;
constexpr std::size_t max_iters = 10;

const std::size_t nsteps = 30;
#define KKTFILEFORMAT "{}.{:03d}.npy"

struct extract_kkt_matrix_callback : aligator::CallbackBaseTpl<double> {
  std::string filepath;
  extract_kkt_matrix_callback(std::string const &filepath)
      : filepath(filepath) {}
  void call(const Workspace &ws_, const Results &) {
    const auto &ws = static_cast<const aligator::context::Workspace &>(ws_);
    for (std::size_t t = 0; t < ws.kkt_mats_.size(); t++) {
      MatrixXd const &w = ws.kkt_mats_[t];
      auto fname = fmt::format(KKTFILEFORMAT, filepath, t);
      cnpy::npy_save_mat(fname, w);
      auto fp2 = fmt::format(KKTFILEFORMAT, "kkt_vecs", t);
      cnpy::npy_save_mat(fp2, ws.kkt_rhs_[t]);
    }
  }
};

int main(int, char **) {

  auto croc_problem = defineCrocoddylProblem(nsteps);
  auto problem = aligator::compat::croc::convertCrocoddylProblem(croc_problem);

  double mu_init = 0.01;
  SolverProxDDP<double> solver(TOL, mu_init, 0., max_iters, aligator::VERBOSE);
  std::string fpath_base = "kkt_matrices";
  solver.registerCallback(
      "kktcb", std::make_shared<extract_kkt_matrix_callback>(fpath_base));

  std::vector<VectorXd> xs_i, us_i;
  getInitialGuesses(croc_problem, xs_i, us_i);

  solver.setup(problem);
  solver.run(problem, xs_i, us_i);

  auto &res = solver.results_;
  fmt::print("Results: {}\n", res);
}
