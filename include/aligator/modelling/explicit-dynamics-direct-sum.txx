/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/context.hpp"
#include "./explicit-dynamics-direct-sum.hpp"

namespace aligator {

extern template struct DirectSumExplicitDynamicsTpl<context::Scalar>;

} // namespace aligator
