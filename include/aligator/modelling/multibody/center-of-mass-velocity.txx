#pragma once

#include "proxddp/context.hpp"
#include "proxddp/modelling/multibody/center-of-mass-velocity.hpp"

namespace aligator {

extern template struct CenterOfMassVelocityResidualTpl<context::Scalar>;
extern template struct CenterOfMassVelocityDataTpl<context::Scalar>;

} // namespace aligator
