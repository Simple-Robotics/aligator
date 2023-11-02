/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/context.hpp"
#include "./cost-direct-sum.hpp"

namespace aligator {

extern template struct DirectSumCostTpl<context::Scalar>;

} // namespace aligator
