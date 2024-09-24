/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#include "aligator/modelling/dynamics/integrator-midpoint.hxx"

namespace aligator::dynamics {
template struct IntegratorMidpointTpl<context::Scalar>;
template struct IntegratorMidpointDataTpl<context::Scalar>;
} // namespace aligator::dynamics
