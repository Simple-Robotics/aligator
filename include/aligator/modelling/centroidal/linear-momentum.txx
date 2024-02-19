#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/centroidal/linear-momentum.hpp"

namespace aligator {

extern template struct LinearMomentumResidualTpl<context::Scalar>;
extern template struct LinearMomentumDataTpl<context::Scalar>;

} // namespace aligator
