/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#ifdef ALIGATOR_WITH_PINOCCHIO
#include "aligator/modelling/dynamics/multibody-free-fwd.hxx"

namespace aligator::dynamics {
template struct MultibodyFreeFwdDynamicsTpl<context::Scalar>;
template struct MultibodyFreeFwdDataTpl<context::Scalar>;
} // namespace aligator::dynamics
#endif
