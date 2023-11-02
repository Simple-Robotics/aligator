#pragma once

#include "./quad-state-cost.hpp"

namespace aligator {

extern template struct QuadraticStateCostTpl<context::Scalar>;
extern template struct QuadraticControlCostTpl<context::Scalar>;

} // namespace aligator
