#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/centroidal/centroidal-friction-cone.hpp"

namespace aligator {

extern template struct CentroidalFrictionConeResidualTpl<context::Scalar>;
extern template struct CentroidalFrictionConeDataTpl<context::Scalar>;

} // namespace aligator
