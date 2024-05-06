#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/multibody/centroidal-momentum.hpp"

namespace aligator {

extern template struct CentroidalMomentumResidualTpl<context::Scalar>;
extern template struct CentroidalMomentumDataTpl<context::Scalar>;

} // namespace aligator
