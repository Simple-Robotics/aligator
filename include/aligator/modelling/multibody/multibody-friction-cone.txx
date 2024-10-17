#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/multibody/multibody-friction-cone.hpp"

namespace aligator {

extern template struct MultibodyFrictionConeResidualTpl<context::Scalar>;
extern template struct MultibodyFrictionConeDataTpl<context::Scalar>;

} // namespace aligator
