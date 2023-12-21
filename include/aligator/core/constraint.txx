/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/context.hpp"
#include "aligator/core/constraint.hpp"

namespace aligator {

extern template struct StageConstraintTpl<context::Scalar>;

extern template struct ConstraintStackTpl<context::Scalar>;

} // namespace aligator
