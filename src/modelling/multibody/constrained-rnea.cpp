#include "proxddp/fwd.hpp"
#ifdef PROXDDP_PINOCCHIO_V3
#include "proxddp/modelling/multibody/constrained-rnea.hpp"

namespace proxddp {

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

} // namespace proxddp
#endif
