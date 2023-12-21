#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/multibody/center-of-mass-translation.hpp"

namespace aligator {

extern template struct CenterOfMassTranslationResidualTpl<context::Scalar>;
extern template struct CenterOfMassTranslationDataTpl<context::Scalar>;

} // namespace aligator
