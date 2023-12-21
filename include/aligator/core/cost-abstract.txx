/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/context.hpp"

namespace aligator {

extern template struct CostAbstractTpl<context::Scalar>;

extern template struct CostDataAbstractTpl<context::Scalar>;

} // namespace aligator
