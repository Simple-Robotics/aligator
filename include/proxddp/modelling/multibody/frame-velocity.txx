#pragma once

#include "proxddp/context.hpp"
#include "proxddp/modelling/multibody/frame-velocity.hpp"

namespace proxddp {

extern template struct FrameVelocityResidualTpl<context::Scalar>;
extern template struct FrameVelocityDataTpl<context::Scalar>;

} // namespace proxddp
