/// @copyright Copyright (C) 2024 LAAS-CNRS, 2024-2025 INRIA
#include "aligator/modelling/dynamics/integrator-explicit.hpp"

namespace aligator {
namespace dynamics {

template struct ExplicitIntegratorAbstractTpl<context::Scalar>;
template struct ExplicitIntegratorDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
