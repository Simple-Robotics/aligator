/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
/// @author Wilson Jallet
#pragma once

#ifdef ALIGATOR_WITH_CHOLMOD
#include <Eigen/CholmodSupport>

#include "lqr-problem.hpp"
#include "aligator/context.hpp"

namespace aligator::gar {
namespace helpers {
/// @brief Helper to assign a dense matrix into a range of coefficients of a
/// sparse matrix.
template <typename InType, typename OutScalar>
void sparseAssignDenseBlock(Eigen::Index i0, Eigen::Index j0,
                            const Eigen::DenseBase<InType> &input,
                            Eigen::SparseMatrix<OutScalar> &out, bool update) {
  assert(i0 + input.rows() <= out.rows() && "Inconsistent rows");
  assert(j0 + input.cols() <= out.cols() && "Inconsistent cols");
  using Eigen::Index;
  for (Index i = 0; i < input.rows(); i++) {
    for (Index j = 0; j < input.cols(); j++) {
      if (update)
        out.coeffRef(i0 + i, j0 + j) = input(i, j);
      else
        out.insert(i0 + i, j0 + j) = input(i, j);
    }
  }
}

template <typename Scalar>
void sparseAssignDiagonal(Eigen::Index i0, Eigen::Index i1, Scalar value,
                          Eigen::SparseMatrix<Scalar> &out, bool update) {
  using Eigen::Index;
  assert(i0 <= i1 && "i0 should be lesser than i1. Can't assign empty range.");
  assert(i1 <= out.rows() && "Inconsistent rows");
  assert(i1 <= out.cols() && "Inconsistent cols");
  for (Index kk = i0; kk < i1; kk++) {
    if (update)
      out.coeffRef(kk, kk) = value;
    else
      out.insert(kk, kk) = value;
  }
}
} // namespace helpers

template <typename Scalar>
void lqrCreateSparseMatrix(const LQRProblemTpl<Scalar> &problem,
                           const Scalar mudyn, const Scalar mueq,
                           Eigen::SparseMatrix<Scalar> &mat,
                           Eigen::Matrix<Scalar, -1, 1> &rhs, bool update);

/// @brief A sparse solver for the linear-quadratic problem based on CHOLMOD.
template <typename _Scalar> class CholmodLqSolver {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Problem = LQRProblemTpl<Scalar>;
  using SparseType = Eigen::SparseMatrix<Scalar>;
  using Triplet = Eigen::Triplet<Scalar>;

  explicit CholmodLqSolver(const Problem &problem, uint numRefinementSteps = 1);

  bool backward(const Scalar mudyn, const Scalar mueq);

  bool forward(std::vector<VectorXs> &xs, std::vector<VectorXs> &us,
               std::vector<VectorXs> &vs, std::vector<VectorXs> &lbdas) const;

  inline Scalar computeSparseResidual() const {
    kktResidual = kktRhs;
    kktResidual.noalias() += kktMatrix * kktSol;
    return math::infty_norm(kktResidual);
  }

  /// Linear problem matrix
  SparseType kktMatrix;
  /// Linear problem rhs
  VectorXs kktRhs;
  /// KKT problem residual
  mutable VectorXs kktResidual;
  /// Linear problem solution
  mutable VectorXs kktSol;
  Eigen::CholmodSimplicialLDLT<SparseType> cholmod;
  /// Number of iterative refinement steps.
  uint numRefinementSteps;

protected:
  const Problem *problem_;
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template void lqrCreateSparseMatrix<context::Scalar>(
    const LQRProblemTpl<context::Scalar> &problem, const context::Scalar mudyn,
    const context::Scalar mueq, Eigen::SparseMatrix<context::Scalar> &mat,
    context::VectorXs &rhs, bool update);
extern template class CholmodLqSolver<context::Scalar>;
#endif

} // namespace aligator::gar
#endif
