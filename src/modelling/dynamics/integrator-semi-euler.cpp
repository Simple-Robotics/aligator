/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#include "aligator/modelling/dynamics/integrator-semi-euler.hxx"

namespace aligator::dynamics {

template struct IntegratorSemiImplEulerTpl<context::Scalar>;
template struct IntegratorSemiImplDataTpl<context::Scalar>;

} // namespace aligator::dynamics
