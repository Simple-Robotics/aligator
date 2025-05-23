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
    const StdVectorEigenAligned<context::RCM> &,
    StdVectorEigenAligned<context::RCD> &,
    const Eigen::MatrixBase<context::VectorRef> &);

} // namespace aligator
