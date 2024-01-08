/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/dynamics/integrator-abstract.hpp"

namespace aligator {
namespace dynamics {

extern template struct IntegratorAbstractTpl<context::Scalar>;
extern template struct IntegratorDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
