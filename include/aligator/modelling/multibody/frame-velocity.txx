#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/multibody/frame-velocity.hpp"

namespace aligator {

extern template struct FrameVelocityResidualTpl<context::Scalar>;
extern template struct FrameVelocityDataTpl<context::Scalar>;

} // namespace aligator
