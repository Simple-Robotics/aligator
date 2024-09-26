#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/centroidal/centroidal-wrench-cone.hpp"

namespace aligator {

extern template struct CentroidalWrenchConeResidualTpl<context::Scalar>;
extern template struct CentroidalWrenchConeDataTpl<context::Scalar>;

} // namespace aligator
