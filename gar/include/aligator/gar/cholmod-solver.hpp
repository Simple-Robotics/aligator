/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
/// @author Wilson Jallet
#pragma once

#include <Eigen/CholmodSupport>

#include "lqr-problem.hpp"
#include "utils.hpp"
#include "aligator/context.hpp"

namespace aligator::gar {
namespace helpers {
/// @brief Helper to assign a dense matrix into a range of coefficients of a
/// sparse matrix.
template <bool Update, typename InType, typename OutScalar>
void sparseAssignDenseBlock(Eigen::Index i0, Eigen::Index j0,
                            const Eigen::DenseBase<InType> &input,
                            Eigen::SparseMatrix<OutScalar> &out) {
  assert(i0 + input.rows() <= out.rows() && "Inconsistent rows");
  assert(j0 + input.cols() <= out.cols() && "Inconsistent cols");
  using Eigen::Index;
  for (Index i = 0; i < input.rows(); i++) {
    for (Index j = 0; j < input.cols(); j++) {
      if constexpr (Update) {
        out.coeffRef(i0 + i, j0 + j) = input(i, j);
      } else {
        out.insert(i0 + i, j0 + j) = input(i, j);
      }
    }
  }
}
} // namespace helpers

template <bool Update, typename Scalar>
void lqrCreateSparseMatrix(const LQRProblemTpl<Scalar> &problem,
                           const Scalar mudyn, const Scalar mueq,
                           Eigen::SparseMatrix<Scalar> &mat,
                           Eigen::Matrix<Scalar, -1, 1> &rhs) {
  using Eigen::Index;
  const uint nrows = lqrNumRows(problem);
  using knot_t = LQRKnotTpl<Scalar>;
  const auto &knots = problem.stages;

  if constexpr (!Update) {
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
    helpers::sparseAssignDenseBlock<Update>(0, nc0, problem.G0, mat);
    helpers::sparseAssignDenseBlock<Update>(nc0, 0, problem.G0.transpose(),
                                            mat);
    for (Index kk = 0; kk < nc0; kk++) {
      if constexpr (Update) {
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
    helpers::sparseAssignDenseBlock<Update>(idx, idx, model.Q, mat);
    const Index i0 = idx + model.nx; // u
    // S block
    helpers::sparseAssignDenseBlock<Update>(i0, idx, model.S.transpose(), mat);
    helpers::sparseAssignDenseBlock<Update>(idx, i0, model.S, mat);
    // R block
    helpers::sparseAssignDenseBlock<Update>(i0, i0, model.R, mat);

    const Index i1 = i0 + model.nu; // v
    // C block
    helpers::sparseAssignDenseBlock<Update>(i1, idx, model.C, mat);
    helpers::sparseAssignDenseBlock<Update>(idx, i1, model.C.transpose(), mat);
    // D block
    helpers::sparseAssignDenseBlock<Update>(i1, i0, model.D, mat);
    helpers::sparseAssignDenseBlock<Update>(i0, i1, model.D.transpose(), mat);

    const Index i2 = i1 + model.nc;
    // dual block
    for (Index kk = i1; kk < i2; kk++) {
      if constexpr (Update) {
        mat.coeffRef(kk, kk) = -mueq;
      } else {
        mat.insert(kk, kk) = -mueq;
      }
    }

    if (t != N) {
      rhs.segment(idx + n, model.nx2) = model.f;

      // A
      helpers::sparseAssignDenseBlock<Update>(i2, idx, model.A, mat);
      helpers::sparseAssignDenseBlock<Update>(idx, i2, model.A.transpose(),
                                              mat);
      // B
      helpers::sparseAssignDenseBlock<Update>(i2, i0, model.B, mat);
      helpers::sparseAssignDenseBlock<Update>(i0, i2, model.B.transpose(), mat);
      // E
      const Index i3 = i2 + model.nx2;
      helpers::sparseAssignDenseBlock<Update>(i2, i3, model.E, mat);
      helpers::sparseAssignDenseBlock<Update>(i3, i2, model.E.transpose(), mat);

      for (Index kk = i2; kk < i3; kk++) {
        if constexpr (Update) {
          mat.coeffRef(kk, kk) = -mudyn;
        } else {
          mat.insert(kk, kk) = -mudyn;
        }
      }

      idx += n + model.nx2;
    }
  }
}

/// @brief A sparse solver for the linear-quadratic problem based on CHOLMOD.
template <typename _Scalar> class CholmodLqSolver {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Problem = LQRProblemTpl<Scalar>;
  using SparseType = Eigen::SparseMatrix<Scalar>;
  using Triplet = Eigen::Triplet<Scalar>;

  explicit CholmodLqSolver(const Problem &problem);

  bool backward(const Scalar mudyn, const Scalar mueq) {
    // update the sparse linear problem
    lqrCreateSparseMatrix<true>(*problem_, mudyn, mueq, kktMatrix, kktRhs);
    cholmod.factorize(kktMatrix);
    return cholmod.info() == Eigen::Success;
  }

  bool forward(std::vector<VectorXs> &xs, std::vector<VectorXs> &us,
               std::vector<VectorXs> &vs, std::vector<VectorXs> &lbdas) const {
    kktSol = cholmod.solve(-kktRhs);
    lqrDenseSolutionToTraj(*problem_, kktSol, xs, us, vs, lbdas);
    return true;
  };

  Scalar computeSparseResidual() {
    kktRhs.noalias() += kktMatrix * kktSol;
    return math::infty_norm(kktRhs);
  }

  /// Linear problem matrix
  SparseType kktMatrix;
  /// Linear problem rhs
  VectorXs kktRhs;
  /// Linear problem solution
  mutable VectorXs kktSol;
  Eigen::CholmodSimplicialLDLT<SparseType> cholmod;

protected:
  const Problem *problem_;
};

template <typename Scalar>
CholmodLqSolver<Scalar>::CholmodLqSolver(const Problem &problem)
    : kktMatrix(), kktRhs(), cholmod(), problem_(&problem) {
  lqrCreateSparseMatrix<false>(problem, 1., 1., kktMatrix, kktRhs);
  assert(kktMatrix.cols() == kktRhs.rows());
  kktSol.resize(kktRhs.rows());
  cholmod.analyzePattern(kktMatrix);
}

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template class CholmodLqSolver<context::Scalar>;
#endif

} // namespace aligator::gar
