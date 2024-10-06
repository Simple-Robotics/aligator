/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/context.hpp"
#include "aligator/core/dynamics.hpp"

namespace aligator {

extern template struct DynamicsModelTpl<context::Scalar>;
extern template struct DynamicsDataTpl<context::Scalar>;

} // namespace aligator
