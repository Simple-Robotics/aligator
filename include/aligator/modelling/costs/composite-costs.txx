/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/costs/composite-costs.hpp"

namespace aligator {
extern template struct CompositeCostDataTpl<context::Scalar>;
} // namespace aligator
