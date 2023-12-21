#pragma once

#include "proxddp/context.hpp"
#include "proxddp/modelling/multibody/center-of-mass-translation.hpp"

namespace aligator {

extern template struct CenterOfMassTranslationResidualTpl<context::Scalar>;
extern template struct CenterOfMassTranslationDataTpl<context::Scalar>;

} // namespace aligator
