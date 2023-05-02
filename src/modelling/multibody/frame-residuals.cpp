#include "proxddp/modelling/multibody/frame-placement.hpp"
#include "proxddp/modelling/multibody/frame-translation.hpp"
#include "proxddp/modelling/multibody/frame-velocity.hpp"

namespace proxddp {

template struct FramePlacementResidualTpl<context::Scalar>;
template struct FramePlacementDataTpl<context::Scalar>;

template struct FrameTranslationResidualTpl<context::Scalar>;
template struct FrameTranslationDataTpl<context::Scalar>;

template struct FrameVelocityResidualTpl<context::Scalar>;
template struct FrameVelocityDataTpl<context::Scalar>;

} // namespace proxddp
