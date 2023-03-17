#pragma once

#include "proxddp/context.hpp"
#include "proxddp/modelling/dynamics/ode-abstract.hpp"

namespace proxddp {
namespace dynamics {

extern template struct ODEAbstractTpl<context::Scalar>;
extern template struct ODEDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace proxddp
