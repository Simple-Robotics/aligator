/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#include "aligator/modelling/dynamics/continuous-dynamics-abstract.hxx"
#include "aligator/core/manifold-base.hpp"

namespace aligator {
namespace dynamics {

template struct ContinuousDynamicsAbstractTpl<context::Scalar>;
template struct ContinuousDynamicsDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
