/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#include "aligator/gar/utils.hxx"

namespace aligator {
namespace gar {
template void lqrCreateSparseMatrix<context::Scalar>(
    const LqrProblemTpl<context::Scalar> &problem, const context::Scalar mueq,
    Eigen::SparseMatrix<context::Scalar> &mat, context::VectorXs &rhs,
    bool update);
template std::array<context::Scalar, 3> lqrComputeKktError<context::Scalar>(
    const LqrProblemTpl<context::Scalar> &,
    boost::span<const context::VectorXs>, boost::span<const context::VectorXs>,
    boost::span<const context::VectorXs>, boost::span<const context::VectorXs>,
    const context::Scalar, const context::Scalar,
    const std::optional<context::ConstVectorRef> &, bool);
template auto
lqrDenseMatrix<context::Scalar>(const LqrProblemTpl<context::Scalar> &,
                                const context::Scalar);
} // namespace gar
} // namespace aligator
