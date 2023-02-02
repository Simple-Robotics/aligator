/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/context.hpp"
#include "proxddp/core/constraint.hpp"

namespace proxddp {

extern template struct StageConstraintTpl<context::Scalar>;

extern template struct ConstraintStackTpl<context::Scalar>;

}
