/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#include "aligator/modelling/dynamics/integrator-abstract.hxx"

namespace aligator {
namespace dynamics {

template struct IntegratorAbstractTpl<context::Scalar>;
template struct IntegratorDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
