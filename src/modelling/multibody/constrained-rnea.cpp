#include "aligator/fwd.hpp"

#include "aligator/modelling/multibody/constrained-rnea.hpp"

namespace aligator {

template void underactuatedConstrainedInverseDynamics<
    context::Scalar, context::ConstVectorRef, context::ConstVectorRef,
    context::ConstMatrixRef, context::VectorRef, context::Options>(
    const context::PinModel &, context::PinData &,
    const Eigen::MatrixBase<context::ConstVectorRef> &,
    const Eigen::MatrixBase<context::ConstVectorRef> &,
    const Eigen::MatrixBase<context::ConstMatrixRef> &,
    const PINOCCHIO_ALIGNED_STD_VECTOR(context::RCM) &,
    PINOCCHIO_ALIGNED_STD_VECTOR(context::RCD) &,
    const Eigen::MatrixBase<context::VectorRef> &);

} // namespace aligator
