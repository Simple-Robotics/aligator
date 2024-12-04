#ifdef ALIGATOR_PINOCCHIO_V3
#include "aligator/modelling/multibody/multibody-friction-cone.hxx"

namespace aligator {

template struct MultibodyFrictionConeResidualTpl<context::Scalar>;
template struct MultibodyFrictionConeDataTpl<context::Scalar>;

} // namespace aligator

#endif
