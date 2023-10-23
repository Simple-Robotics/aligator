#include "croc-talos-arm.hpp"

#include "proxddp/core/solver-proxddp.hpp"
#include <proxnlp/tests/cnpy.hpp>
#include <fmt/ostream.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

using proxddp::SolverProxDDP;

constexpr double TOL = 1e-4;
constexpr std::size_t max_iters = 10;

const std::size_t nsteps = 30;
#define KKTFILEFORMAT "{}.{:03d}.npy"

struct extract_kkt_matrix_callback : proxddp::CallbackBaseTpl<double> {
  std::string filepath;
  extract_kkt_matrix_callback(std::string const &filepath)
      : filepath(filepath) {}
  void call(const Workspace &ws_, const Results &) {
    const auto &ws = static_cast<const proxddp::context::Workspace &>(ws_);
    for (std::size_t t = 0; t < ws.kkt_mats_.size(); t++) {
      MatrixXd const &w = ws.kkt_mats_[t];
      auto fname = fmt::format(KKTFILEFORMAT, filepath, t);
      cnpy::npy_save_mat(fname, w);
      auto fp2 = fmt::format(KKTFILEFORMAT, "kkt_vecs", t);
      cnpy::npy_save_mat(fp2, ws.kkt_rhs_[t]);
    }
  }
};

template <typename Algo, typename MatType>
auto compute_reconstr_err(Algo &chol, MatType &mat) {
  return proxddp::math::infty_norm(chol.reconstructedMatrix() - mat);
}

template <typename Algo, typename MatType, typename Rhs>
auto compute_solve_err(Algo &chol, MatType &mat, Rhs &rhs) {
  auto sol = rhs;
  chol.solveInPlace(sol);
  return proxnlp::math::infty_norm(rhs - (mat * sol));
}

void factorize_some_kkt(std::string &fname, std::string &frhs) {
  using proxnlp::linalg::BlockLDLT;
  using proxnlp::linalg::DenseLDLT;

  auto mat = cnpy::npy_load_mat<double>(fname);
  auto rhs = cnpy::npy_load_mat<double>(frhs);

  // dense ldlt
  DenseLDLT<double> dense_ldlt(mat);
  double dense_err = compute_reconstr_err(dense_ldlt, mat);
  fmt::print("DenseLDLT: {:.4e}", dense_err);
  fmt::print(" /solve: {:.4e}\n", compute_solve_err(dense_ldlt, mat, rhs));

  Eigen::LDLT<MatrixXd> eigen_ldlt(mat);
  double eig_err = compute_reconstr_err(eigen_ldlt, mat);
  fmt::print("EigenLDLT: {:.4e}", eig_err);
  fmt::print(" /solve: {:.4e}\n", compute_solve_err(eigen_ldlt, mat, rhs));

  auto nprims = {7L, 14L};
  auto nduals = {14L};
  proxddp::unique_ptr<BlockLDLT<double>> p_blk_chol(
      proxddp::custom_block_ldlt_allocator<double>::create(nprims, nduals,
                                                           true));
  auto &block_ldlt = *p_blk_chol;

  long perm[3] = {1, 2, 0};
  block_ldlt.setBlockPermutation(perm);

  block_ldlt.compute(mat);
  double block_err = compute_reconstr_err(block_ldlt, mat);
  fmt::print("BlockLDLT: {:.4e}", block_err);
  fmt::print(" /solve: {:.4e}\n", compute_solve_err(block_ldlt, mat, rhs));

  //
}

int main(int, char **) {

  auto croc_problem = defineCrocoddylProblem(nsteps);
  auto problem = proxddp::compat::croc::convertCrocoddylProblem(croc_problem);

  double mu_init = 0.01;
  SolverProxDDP<double> solver(TOL, mu_init, 0., max_iters, proxddp::VERBOSE);
  std::string fpath_base = "kkt_matrices";
  solver.registerCallback(
      "kktcb", std::make_shared<extract_kkt_matrix_callback>(fpath_base));

  std::vector<VectorXd> xs_i, us_i;
  getInitialGuesses(croc_problem, xs_i, us_i);

  solver.setup(problem);
  solver.run(problem, xs_i, us_i);

  auto &res = solver.results_;
  fmt::print("Results: {}\n", res);

  for (std::size_t i = 1; i < nsteps; ++i) {
    auto fpath = fmt::format(KKTFILEFORMAT, fpath_base, i);
    fmt::print("Facto kkt from file {}\n", fpath);
    auto fp2vec = fmt::format(KKTFILEFORMAT, "kkt_vecs", i);
    factorize_some_kkt(fpath, fp2vec);
  }
}
