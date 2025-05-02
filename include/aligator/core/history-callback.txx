#pragma once

#include "aligator/context.hpp"
#include "./history-callback.hpp"

namespace aligator {

extern template struct HistoryCallbackTpl<context::Scalar>;

} // namespace aligator
