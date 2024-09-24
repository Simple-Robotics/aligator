/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#include "aligator/fwd.hpp"

#ifdef ALIGATOR_PINOCCHIO_V3
#include "aligator/modelling/dynamics/multibody-constraint-fwd.hxx"

namespace aligator::dynamics {

template struct MultibodyConstraintFwdDynamicsTpl<context::Scalar>;
template struct MultibodyConstraintFwdDataTpl<context::Scalar>;

} // namespace aligator::dynamics
#endif
