/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#include "aligator/modelling/dynamics/continuous-dynamics-abstract.hpp"
#include <proxsuite-nlp/manifold-base.hpp>

namespace aligator {
namespace dynamics {

template struct ContinuousDynamicsAbstractTpl<context::Scalar>;
template struct ContinuousDynamicsDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
