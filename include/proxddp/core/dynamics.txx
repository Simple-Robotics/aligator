/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/context.hpp"
#include "proxddp/core/dynamics.hpp"

namespace proxddp {

extern template struct DynamicsModelTpl<context::Scalar>;

} // namespace proxddp
