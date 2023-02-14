#pragma once

#include "proxddp/context.hpp"

#include "proxddp/modelling/quad-costs.hpp"

namespace proxddp {

extern template struct QuadraticCostTpl<context::Scalar>;
extern template struct QuadraticCostDataTpl<context::Scalar>;

}
