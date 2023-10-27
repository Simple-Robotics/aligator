#pragma once

#include "proxddp/context.hpp"

namespace proxddp {

extern template struct StageFunctionTpl<context::Scalar>;
extern template struct StageFunctionDataTpl<context::Scalar>;

} // namespace proxddp
