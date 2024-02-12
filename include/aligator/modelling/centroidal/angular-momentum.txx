#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/centroidal/angular-momentum.hpp"

namespace aligator {

extern template struct AngularMomentumResidualTpl<context::Scalar>;
extern template struct AngularMomentumDataTpl<context::Scalar>;

} // namespace aligator
