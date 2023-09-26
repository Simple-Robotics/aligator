/// Implementation of a parallel LQR algorithm
#pragma once

#include "aligator/context.hpp"
#include "aligator/math.hpp"

#include "./factor.hpp"
#include "./lqr-knot.hpp"
#include "./lqr-tree.hpp"
#include "./util.hpp"

#include <Eigen/Cholesky>

namespace aligator {

template <typename T> class LQRTreeSolver {
public:
  PROXNLP_DYNAMIC_TYPEDEFS(T);
  struct ldlt_t {
    MatrixXs mat;
    Eigen::LDLT<MatrixXs> fac;
    ldlt_t(size_t dim) : mat(dim, dim), fac(mat) {}
  };
  using factor_t = LQRFactor<T>;
  using problem_t = LQRProblem<T>;
  using knot_t = LQRKnot<T>;

  size_t horz;
  LQRTree tree;
  problem_t const *problem;

  // right hand sides and also solution
  std::vector<factor_t> rhsAndSol;
  // inner matrix factors
  std::vector<factor_t> factors;
  // factorization for the leaf nodes
  std::vector<ldlt_t> ldlts;

  LQRTreeSolver(const problem_t &problem)
      : horz(problem.horizon()), tree(problem.horizon()), problem(&problem) {

    size_t depth = tree.maxDepth();
    for (size_t i = 0; i < horz; i++) {
      // total dim
      const knot_t &stage = problem.stages[i];
      auto nx = stage.nx;
      auto nu = stage.nu;
      auto nc = stage.nc;
      const auto dim = nx + nu + nc;

      // allocate LDLT solver memory
      ldlts.emplace_back(dim);
      MatrixXs &mat = ldlts[i].mat;
      mat.block(0, 0, nx, nx) = stage.Q;
      mat.block(nx, 0, nu, nx) = stage.S.transpose();
      mat.block(nx, nx, nu, nu) = stage.R;
      mat.block(nx + nu, 0, nc, nx) = stage.C;
      mat.block(nx + nu, nx, nc, nx) = stage.D;
      mat = mat.template selfadjointView<Eigen::Lower>();

      // allocate rhs and fill in
      rhsAndSol.emplace_back(nx, nu, nc, 1); // width of 1
      factor_t &rhs = rhsAndSol[i];
      rhs.X = stage.q;
      rhs.U = stage.r;

      // memory layout of factors is row-major:
      // column index is depth, row index is horizon
      for (size_t j = 0; j < depth; j++) {
        factors.emplace_back(nx, nu, nc, nx);
      }

      size_t idx = i * depth + depth - 1;
      factor_t &fac = factors[idx];
      fac.X = stage.A.transpose();
      fac.U = stage.B.transpose();
    }
  }

  /// Solve the i-th leaf.
  void solveLeaf(size_t i) {
    const knot_t &stage = problem->stages[i];
    auto nx = stage.nx;
    auto nu = stage.nu;
    auto nc = stage.nc;

    ldlt_t &ldlt = ldlts[i];
    ldlt.fac.compute(ldlt.mat); // ldlt.mat already set, factorize

    // solve for right hand side
    factor_t &rhs = rhsAndSol[i];
    // solve for X, U, Nu
    auto rhs_xun = rhs.data.topRows(nx + nu + nc);
    ldlt.fac.solveInPlace(rhs_xun);

    factor_t &fac = factors[i];
    auto dm = fac.data.topRows(nx + nu + nc);
    ldlt.fac.solveInPlace(dm);
  }

  void solve() {

    for (size_t i = 0; i < horz; ++i) {
      solveLeaf(i);
    }
  }
};

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
extern template struct LQRKnot<context::Scalar>;
extern template struct LQRFactor<context::Scalar>;
extern template class LQRTreeSolver<context::Scalar>;
#endif
} // namespace aligator
