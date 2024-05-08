/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
/// @author Wilson Jallet
#pragma once

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

  explicit CholmodLqSolver(const Problem &problem);
  void buildProblemSparseMatrix(const LQRProblemTpl<Scalar> &problem,
                                std::vector<Triplet> &coefficients) {
    coefficients;
    kktMatrix_.setFromTriplets(coefficients.begin(), coefficients.end());
  }
  void addProblemRegularization(const Scalar mudyn, const Scalar mueq) {}

  void backward() {}

protected:
  /// Linear problem matrix
  SparseType kktMatrix_;
  Eigen::CholmodSimplicialLDLT<SparseType> cholmod;
  const Problem *problem_;
};

template <typename Scalar>
CholmodLqSolver<Scalar>::CholmodLqSolver(const Problem &problem)
    : cholmod(), problem_(&problem) {}

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template class CholmodLqSolver<context::Scalar>;
#endif

} // namespace aligator::gar
