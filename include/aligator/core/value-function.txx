#pragma once
#include "aligator/context.hpp"
#include "aligator/core/value-function.hpp"

namespace aligator {
extern template struct ValueFunctionTpl<context::Scalar>;
extern template struct QFunctionTpl<context::Scalar>;
} // namespace aligator
