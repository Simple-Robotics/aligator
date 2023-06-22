#pragma once
#include "proxddp/context.hpp"
#include "proxddp/core/value-function.hpp"

namespace proxddp {
  extern template struct ValueFunctionTpl<context::Scalar>;
  extern template struct QFunctionTpl<context::Scalar>;
}
