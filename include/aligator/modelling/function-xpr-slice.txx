/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/context.hpp"
#include "./function-xpr-slice.hpp"

namespace aligator {

extern template struct FunctionSliceXprTpl<context::Scalar,
                                           context::StageFunction>;
extern template struct FunctionSliceXprTpl<context::Scalar,
                                           context::UnaryFunction>;

} // namespace aligator
