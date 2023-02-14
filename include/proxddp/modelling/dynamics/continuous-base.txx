/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/context.hpp"
#include "proxddp/modelling/dynamics/continuous-base.hpp"

namespace proxddp {
namespace dynamics {

extern template struct ContinuousDynamicsAbstractTpl<context::Scalar>;
extern template struct ContinuousDynamicsDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace proxddp
