#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/multibody/multibody-wrench-cone.hpp"

namespace aligator {

extern template struct MultibodyWrenchConeResidualTpl<context::Scalar>;
extern template struct MultibodyWrenchConeDataTpl<context::Scalar>;

} // namespace aligator
