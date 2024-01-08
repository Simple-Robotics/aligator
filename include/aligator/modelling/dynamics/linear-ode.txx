/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/dynamics/linear-ode.hpp"

namespace aligator {
namespace dynamics {

extern template struct LinearODETpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
