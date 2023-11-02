#pragma once

#include "aligator/context.hpp"
#include "./sum-of-costs.hpp"

namespace aligator {

extern template struct CostStackTpl<context::Scalar>;
extern template struct CostStackDataTpl<context::Scalar>;

} // namespace aligator
