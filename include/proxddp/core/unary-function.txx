#pragma once

#include "./unary-function.hpp"
#include "proxddp/context.hpp"

namespace proxddp {

extern template struct UnaryFunctionTpl<context::Scalar>;

} // namespace proxddp
