#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/centroidal/wrench-cone.hpp"

namespace aligator {

extern template struct WrenchConeResidualTpl<context::Scalar>;
extern template struct WrenchConeDataTpl<context::Scalar>;

} // namespace aligator
