#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/multibody/centroidal-momentum-derivative.hpp"

namespace aligator {

extern template struct CentroidalMomentumDerivativeResidualTpl<context::Scalar>;
extern template struct CentroidalMomentumDerivativeDataTpl<context::Scalar>;

} // namespace aligator
