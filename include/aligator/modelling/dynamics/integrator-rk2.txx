/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/dynamics/integrator-rk2.hpp"

namespace aligator {
namespace dynamics {

extern template struct IntegratorRK2Tpl<context::Scalar>;
extern template struct IntegratorRK2DataTpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
