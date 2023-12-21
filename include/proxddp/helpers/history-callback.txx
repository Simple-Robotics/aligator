#pragma once

#include "proxddp/context.hpp"
#include "./history-callback.hpp"

namespace aligator {

extern template struct HistoryCallbackTpl<context::Scalar>;

} // namespace aligator
