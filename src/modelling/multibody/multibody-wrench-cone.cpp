#include "aligator/modelling/multibody/multibody-wrench-cone.hxx"
#ifdef ALIGATOR_PINOCCHIO_V3

namespace aligator {

template struct MultibodyWrenchConeResidualTpl<context::Scalar>;
template struct MultibodyWrenchConeDataTpl<context::Scalar>;

} // namespace aligator

#endif
