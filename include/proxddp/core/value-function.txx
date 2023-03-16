#pragma once
#include "proxddp/context.hpp"
#include "proxddp/core/value-function.hpp"

namespace proxddp {
  extern template struct value_function<context::Scalar>;
  extern template struct q_function<context::Scalar>;
}
