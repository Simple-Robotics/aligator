/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
/// @author Wilson Jallet
#pragma once

#ifdef ALIGATOR_WITH_CHOLMOD
#include "cholmod-solver.hpp"
#include "utils.hpp"

namespace aligator::gar {

template <typename Scalar>
CholmodLqSolver<Scalar>::CholmodLqSolver(const Problem &problem,
                                         uint numRefinementSteps)
    : kktMatrix()
    , kktRhs()
    , cholmod()
    , numRefinementSteps(numRefinementSteps)
    , problem_(&problem) {
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
