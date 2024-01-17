#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/centroidal/centroidal-acceleration.hpp"

namespace aligator {

extern template struct CentroidalAccelerationResidualTpl<context::Scalar>;
extern template struct CentroidalAccelerationDataTpl<context::Scalar>;

} // namespace aligator
