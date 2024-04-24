/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "aligator/gar/utils.hpp"

namespace aligator {
namespace gar {
template auto lqrComputeKktError<context::Scalar>(
    const LQRProblemTpl<context::Scalar> &,
    boost::span<const context::VectorXs>, boost::span<const context::VectorXs>,
    boost::span<const context::VectorXs>, boost::span<const context::VectorXs>,
    const context::Scalar, const context::Scalar,
    const std::optional<context::ConstVectorRef> &, bool);
template auto
lqrDenseMatrix<context::Scalar>(const LQRProblemTpl<context::Scalar> &,
                                context::Scalar, context::Scalar);
} // namespace gar
} // namespace aligator
