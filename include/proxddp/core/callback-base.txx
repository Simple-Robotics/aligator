#pragma once

#include "proxddp/context.hpp"
#include "proxddp/core/callback-base.hpp"

namespace proxddp {
namespace helpers {

extern template struct base_callback<context::Scalar>;

} // namespace helpers
} // namespace proxddp
