#pragma once

#include "proxddp/context.hpp"
#include "proxddp/modelling/multibody/center-of-mass-velocity.hpp"

namespace proxddp {

extern template struct CenterOfMassVelocityResidualTpl<context::Scalar>;
extern template struct CenterOfMassVelocityDataTpl<context::Scalar>;

} // namespace proxddp
