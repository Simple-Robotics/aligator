/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
/// @author Wilson Jallet
#pragma once

#ifdef ALIGATOR_WITH_CHOLMOD
#include <Eigen/CholmodSupport>

#include "lqr-problem.hpp"
#include "aligator/context.hpp"

namespace aligator::gar {

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
extern template class CholmodLqSolver<context::Scalar>;
#endif

} // namespace aligator::gar
#endif
