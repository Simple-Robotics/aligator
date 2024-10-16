/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
/// @author Wilson Jallet
#pragma once

#ifdef ALIGATOR_WITH_CHOLMOD
#include "cholmod-solver.hpp"
#include "utils.hpp"

namespace aligator::gar {

template <typename Scalar>
void lqrCreateSparseMatrix(const LQRProblemTpl<Scalar> &problem,
                           const Scalar mudyn, const Scalar mueq,
                           Eigen::SparseMatrix<Scalar> &mat,
                           Eigen::Matrix<Scalar, -1, 1> &rhs, bool update) {
  using Eigen::Index;
  const uint nrows = lqrNumRows(problem);
  using knot_t = LQRKnotTpl<Scalar>;
  const auto &knots = problem.stages;

  if (!update) {
    mat.conservativeResize(nrows, nrows);
    mat.setZero();
  }

  rhs.conservativeResize(nrows);
  rhs.setZero();
  const size_t N = size_t(problem.horizon());
  uint idx = 0;
  {
    uint nc0 = problem.nc0();
    rhs.head(nc0) = problem.g0;
    helpers::sparseAssignDenseBlock(0, nc0, problem.G0, mat, update);
    helpers::sparseAssignDenseBlock(nc0, 0, problem.G0.transpose(), mat,
                                    update);
    for (Index kk = 0; kk < nc0; kk++) {
      if (update) {
        mat.coeffRef(kk, kk) = -mudyn;
      } else {
        mat.insert(kk, kk) = -mudyn;
      }
    }
    idx += nc0;
  }

  for (size_t t = 0; t <= N; t++) {
    const knot_t &model = knots[t];
    const uint n = model.nx + model.nu + model.nc;
    // get block for current variables
    auto rhsblk = rhs.segment(idx, n);

    rhsblk.head(model.nx) = model.q;
    rhsblk.segment(model.nx, model.nu) = model.r;
    rhsblk.tail(model.nc) = model.d;

    // fill-in Q block
    helpers::sparseAssignDenseBlock(idx, idx, model.Q, mat, update);
    const Index i0 = idx + model.nx; // u
    // S block
    helpers::sparseAssignDenseBlock(i0, idx, model.S.transpose(), mat, update);
    helpers::sparseAssignDenseBlock(idx, i0, model.S, mat, update);
    // R block
    helpers::sparseAssignDenseBlock(i0, i0, model.R, mat, update);

    const Index i1 = i0 + model.nu; // v
    // C block
    helpers::sparseAssignDenseBlock(i1, idx, model.C, mat, update);
    helpers::sparseAssignDenseBlock(idx, i1, model.C.transpose(), mat, update);
    // D block
    helpers::sparseAssignDenseBlock(i1, i0, model.D, mat, update);
    helpers::sparseAssignDenseBlock(i0, i1, model.D.transpose(), mat, update);

    const Index i2 = i1 + model.nc;
    // dual block
    helpers::sparseAssignDiagonal(i1, i2, -mueq, mat, update);

    if (t != N) {
      rhs.segment(idx + n, model.nx2) = model.f;

      // A
      helpers::sparseAssignDenseBlock(i2, idx, model.A, mat, update);
      helpers::sparseAssignDenseBlock(idx, i2, model.A.transpose(), mat,
                                      update);
      // B
      helpers::sparseAssignDenseBlock(i2, i0, model.B, mat, update);
      helpers::sparseAssignDenseBlock(i0, i2, model.B.transpose(), mat, update);
      // E
      const Index i3 = i2 + model.nx2;
      helpers::sparseAssignDenseBlock(i2, i3, model.E, mat, update);
      helpers::sparseAssignDenseBlock(i3, i2, model.E.transpose(), mat, update);

      helpers::sparseAssignDiagonal(i2, i3, -mudyn, mat, update);

      idx += n + model.nx2;
    }
  }
}

template <typename Scalar>
CholmodLqSolver<Scalar>::CholmodLqSolver(const Problem &problem,
                                         uint numRefinementSteps)
    : kktMatrix(), kktRhs(), cholmod(), numRefinementSteps(numRefinementSteps),
      problem_(&problem) {
  lqrCreateSparseMatrix(problem, 1., 1., kktMatrix, kktRhs, false);
  assert(kktMatrix.cols() == kktRhs.rows());
  kktSol.resize(kktRhs.rows());
  kktResidual.resize(kktRhs.rows());
  cholmod.analyzePattern(kktMatrix);
}

template <typename Scalar>
bool CholmodLqSolver<Scalar>::backward(const Scalar mudyn, const Scalar mueq) {
  // update the sparse linear problem
  lqrCreateSparseMatrix(*problem_, mudyn, mueq, kktMatrix, kktRhs, true);
  cholmod.factorize(kktMatrix);
  return cholmod.info() == Eigen::Success;
}

template <typename Scalar>
bool CholmodLqSolver<Scalar>::forward(std::vector<VectorXs> &xs,
                                      std::vector<VectorXs> &us,
                                      std::vector<VectorXs> &vs,
                                      std::vector<VectorXs> &lbdas) const {
  kktSol = cholmod.solve(-kktRhs);
  kktResidual = kktRhs;
  for (uint i = 0; i < numRefinementSteps; i++) {
    kktResidual.noalias() += kktMatrix * kktSol;
    kktSol += cholmod.solve(-kktResidual);
  }
  lqrDenseSolutionToTraj(*problem_, kktSol, xs, us, vs, lbdas);
  return true;
};
} // namespace aligator::gar
#endif
