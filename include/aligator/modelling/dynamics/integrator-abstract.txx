/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#pragma once

#include "./integrator-abstract.hpp"

namespace aligator {
namespace dynamics {

extern template struct IntegratorAbstractTpl<context::Scalar>;
extern template struct IntegratorDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
