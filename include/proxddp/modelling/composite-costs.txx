/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/context.hpp"
#include "./composite-costs.hpp"

namespace aligator {

extern template struct CompositeCostDataTpl<context::Scalar>;

extern template struct QuadraticResidualCostTpl<context::Scalar>;

extern template struct LogResidualCostTpl<context::Scalar>;

} // namespace aligator
