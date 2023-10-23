#pragma once

#include "proxddp/context.hpp"
#include "./history-callback.hpp"

namespace proxddp {

extern template struct HistoryCallbackTpl<context::Scalar>;

} // namespace proxddp
