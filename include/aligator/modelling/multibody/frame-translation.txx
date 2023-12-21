#pragma once

#include "proxddp/context.hpp"
#include "proxddp/modelling/multibody/frame-translation.hpp"

namespace aligator {

extern template struct FrameTranslationResidualTpl<context::Scalar>;
extern template struct FrameTranslationDataTpl<context::Scalar>;

} // namespace aligator
