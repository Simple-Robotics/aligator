#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/centroidal/centroidal-translation.hpp"

namespace aligator {

extern template struct CentroidalCoMResidualTpl<context::Scalar>;
extern template struct CentroidalCoMDataTpl<context::Scalar>;

} // namespace aligator
