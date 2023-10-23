#pragma once

#include "proxddp/context.hpp"
#include "proxddp/core/callback-base.hpp"

namespace proxddp {

extern template struct CallbackBaseTpl<context::Scalar>;

} // namespace proxddp
