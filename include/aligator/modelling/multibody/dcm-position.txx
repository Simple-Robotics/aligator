#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/multibody/dcm-position.hpp"

namespace aligator {

extern template struct DCMPositionResidualTpl<context::Scalar>;
extern template struct DCMPositionDataTpl<context::Scalar>;

} // namespace aligator
