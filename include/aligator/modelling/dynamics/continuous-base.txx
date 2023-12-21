/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/dynamics/continuous-base.hpp"

namespace aligator {
namespace dynamics {

extern template struct ContinuousDynamicsAbstractTpl<context::Scalar>;
extern template struct ContinuousDynamicsDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
