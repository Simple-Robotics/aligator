#pragma once

#include "proxddp/context.hpp"
#include "proxddp/modelling/sum-of-costs.hpp"

namespace aligator {

extern template struct CostStackTpl<context::Scalar>;
extern template struct CostStackDataTpl<context::Scalar>;

} // namespace aligator
