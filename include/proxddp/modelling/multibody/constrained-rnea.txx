#pragma once

#include "./context.hpp"
#include "./constrained-rnea.hpp"

namespace proxddp {

extern template
void underactuatedConstrainedInverseDynamics<context::Scalar, context::ConstVectorRef, context::ConstVectorRef, context::ConstMatrixRef, context::VectorRef, context::Options>(
  const context::PinModel &,
  context::PinData &,
  const Eigen::MatrixBase<context::ConstVectorRef> &,
  const Eigen::MatrixBase<context::ConstVectorRef> &,
  const Eigen::MatrixBase<context::ConstMatrixRef> &,
  const context::RCM &,
  context::RCD &,
  const Eigen::MatrixBase<context::VectorRef> &);

} // namespace proxddp
