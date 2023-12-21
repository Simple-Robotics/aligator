#pragma once

#include "aligator/context.hpp"

#include "aligator/modelling/quad-costs.hpp"

namespace aligator {

extern template struct QuadraticCostTpl<context::Scalar>;
extern template struct QuadraticCostDataTpl<context::Scalar>;

} // namespace aligator
