#pragma once

#include "aligator/modelling/multibody/frame-collision.hpp"
#include "aligator/context.hpp"

namespace aligator {

extern template struct FrameCollisionResidualTpl<context::Scalar>;
extern template struct FrameCollisionDataTpl<context::Scalar>;

} // namespace aligator
