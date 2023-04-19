#pragma once

#include "proxddp/modelling/quad-state-cost.hpp"

namespace proxddp {

extern template struct QuadraticStateCostTpl<context::Scalar>;
extern template struct QuadraticControlCostTpl<context::Scalar>;

} // namespace proxddp
