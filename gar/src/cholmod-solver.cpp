/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#include "aligator/gar/cholmod-solver.hxx"

#ifdef ALIGATOR_WITH_CHOLMOD
namespace aligator::gar {
template void lqrCreateSparseMatrix<context::Scalar>(
    const LQRProblemTpl<context::Scalar> &problem, const context::Scalar mudyn,
    const context::Scalar mueq, Eigen::SparseMatrix<context::Scalar> &mat,
    context::VectorXs &rhs, bool update);
template class CholmodLqSolver<context::Scalar>;
} // namespace aligator::gar
#endif
