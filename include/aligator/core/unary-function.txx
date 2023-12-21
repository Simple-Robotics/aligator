#pragma once

#include "./unary-function.hpp"
#include "aligator/context.hpp"

namespace aligator {

extern template struct UnaryFunctionTpl<context::Scalar>;

} // namespace aligator
