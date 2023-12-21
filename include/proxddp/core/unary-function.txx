#pragma once

#include "./unary-function.hpp"
#include "proxddp/context.hpp"

namespace aligator {

extern template struct UnaryFunctionTpl<context::Scalar>;

} // namespace aligator
