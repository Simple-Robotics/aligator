/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/context.hpp"
#include "proxddp/core/stage-data.hpp"

namespace aligator {

extern template struct StageDataTpl<context::Scalar>;

} // namespace aligator
