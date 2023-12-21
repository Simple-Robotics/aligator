#pragma once

#include "aligator/context.hpp"

namespace aligator {

extern template struct StageFunctionTpl<context::Scalar>;
extern template struct StageFunctionDataTpl<context::Scalar>;

} // namespace aligator
