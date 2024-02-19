#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/centroidal/angular-acceleration.hpp"

namespace aligator {

extern template struct AngularAccelerationResidualTpl<context::Scalar>;
extern template struct AngularAccelerationDataTpl<context::Scalar>;

} // namespace aligator
