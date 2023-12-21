/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/context.hpp"
#include "proxddp/core/results-base.hpp"

namespace aligator {

extern template struct ResultsBaseTpl<context::Scalar>;

} // namespace aligator
