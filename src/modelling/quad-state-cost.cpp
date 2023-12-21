#include "aligator/modelling/quad-state-cost.hpp"

namespace aligator {

template struct QuadraticStateCostTpl<context::Scalar>;
template struct QuadraticControlCostTpl<context::Scalar>;

} // namespace aligator
