#pragma once

#include "proxddp/context.hpp"
#include "proxddp/modelling/dynamics/ode-abstract.hpp"

namespace aligator {
namespace dynamics {

extern template struct ODEAbstractTpl<context::Scalar>;
extern template struct ODEDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
