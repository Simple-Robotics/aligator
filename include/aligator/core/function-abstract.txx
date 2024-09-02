#pragma once

#include "aligator/context.hpp"
#include "aligator/core/function-abstract.hpp"

namespace aligator {

extern template struct StageFunctionTpl<context::Scalar>;
extern template struct StageFunctionDataTpl<context::Scalar>;
extern template std::ostream &
operator<<(std::ostream &oss,
           const StageFunctionDataTpl<context::Scalar> &self);

} // namespace aligator
