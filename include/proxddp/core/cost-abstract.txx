/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/context.hpp"

namespace proxddp {

extern template struct CostAbstractTpl<context::Scalar>;

extern template struct CostDataAbstractTpl<context::Scalar>;

} // namespace proxddp
