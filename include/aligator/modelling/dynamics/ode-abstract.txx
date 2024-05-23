/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/dynamics/ode-abstract.hpp"

namespace aligator {
namespace dynamics {

extern template struct ODEAbstractTpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
