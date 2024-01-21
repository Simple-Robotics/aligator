/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "aligator/gar/utils.hpp"

namespace aligator {
namespace gar {
template auto lqrComputeKktError<context::Scalar>(
    const LQRProblemTpl<context::Scalar> &, const context::VectorOfVectors &,
    const context::VectorOfVectors &, const context::VectorOfVectors &,
    const context::VectorOfVectors &, const context::Scalar,
    const context::Scalar, const std::optional<context::ConstVectorRef> &);
template auto
lqrDenseMatrix<context::Scalar>(const LQRProblemTpl<context::Scalar> &,
                                context::Scalar, context::Scalar);
} // namespace gar
} // namespace aligator
