#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/centroidal/friction-cone.hpp"

namespace aligator {

extern template struct FrictionConeResidualTpl<context::Scalar>;
extern template struct FrictionConeDataTpl<context::Scalar>;

} // namespace aligator
