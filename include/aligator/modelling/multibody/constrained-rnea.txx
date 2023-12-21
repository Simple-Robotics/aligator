#pragma once

#include "./context.hpp"
#include "./constrained-rnea.hpp"
#include "aligator/fwd.hpp"

namespace aligator {

extern template void underactuatedConstrainedInverseDynamics<
    context::Scalar, context::ConstVectorRef, context::ConstVectorRef,
    context::ConstMatrixRef, context::VectorRef, context::Options>(
    const context::PinModel &, context::PinData &,
    const Eigen::MatrixBase<context::ConstVectorRef> &,
    const Eigen::MatrixBase<context::ConstVectorRef> &,
    const Eigen::MatrixBase<context::ConstMatrixRef> &,
    const StdVectorEigenAligned<context::RCM> &,
    StdVectorEigenAligned<context::RCD> &,
    const Eigen::MatrixBase<context::VectorRef> &);

} // namespace aligator
