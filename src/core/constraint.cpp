/// @file
/// @copyright Copyright (C) 2024 LAAS-CNRS, 2024-2025 INRIA
#include "aligator/core/constraint.hpp"

namespace aligator {

template struct StageConstraintTpl<context::Scalar>;

template struct ConstraintStackTpl<context::Scalar>;

} // namespace aligator
