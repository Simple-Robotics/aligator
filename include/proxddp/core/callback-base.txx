#pragma once

#include "proxddp/context.hpp"
#include "proxddp/core/callback-base.hpp"

namespace proxddp {
namespace helpers {

extern template struct CallbackBaseTpl<context::Scalar>;

} // namespace helpers
} // namespace proxddp
