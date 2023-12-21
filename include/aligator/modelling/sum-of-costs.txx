#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/sum-of-costs.hpp"

namespace aligator {

extern template struct CostStackTpl<context::Scalar>;
extern template struct CostStackDataTpl<context::Scalar>;

} // namespace aligator
