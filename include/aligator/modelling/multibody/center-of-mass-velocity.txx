#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/multibody/center-of-mass-velocity.hpp"

namespace aligator {

extern template struct CenterOfMassVelocityResidualTpl<context::Scalar>;
extern template struct CenterOfMassVelocityDataTpl<context::Scalar>;

} // namespace aligator
