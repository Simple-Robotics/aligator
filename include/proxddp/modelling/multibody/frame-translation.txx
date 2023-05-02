#pragma once

#include "proxddp/context.hpp"
#include "proxddp/modelling/multibody/frame-translation.hpp"

namespace proxddp {

extern template struct FrameTranslationResidualTpl<context::Scalar>;
extern template struct FrameTranslationDataTpl<context::Scalar>;

} // namespace proxddp
