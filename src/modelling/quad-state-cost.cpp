#include "proxddp/modelling/quad-state-cost.hpp"

namespace proxddp {

template struct QuadraticStateCostTpl<context::Scalar>;
template struct QuadraticControlCostTpl<context::Scalar>;

} // namespace proxddp
