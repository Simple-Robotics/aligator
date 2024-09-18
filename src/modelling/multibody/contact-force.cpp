#include "aligator/modelling/multibody/contact-force.hxx"
#ifdef ALIGATOR_PINOCCHIO_V3
namespace aligator {

template struct ContactForceResidualTpl<context::Scalar>;
template struct ContactForceDataTpl<context::Scalar>;

} // namespace aligator

#endif
