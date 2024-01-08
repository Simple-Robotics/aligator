/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#include "aligator/modelling/dynamics/integrator-explicit.hpp"

namespace aligator {
namespace dynamics {

template struct ExplicitIntegratorAbstractTpl<context::Scalar>;
template struct ExplicitIntegratorDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
