#pragma once

#include "proxddp/context.hpp"
#include "proxddp/modelling/multibody/frame-velocity.hpp"

namespace aligator {

extern template struct FrameVelocityResidualTpl<context::Scalar>;
extern template struct FrameVelocityDataTpl<context::Scalar>;

} // namespace aligator
