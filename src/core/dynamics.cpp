/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#include "aligator/core/dynamics.hxx"
#include "aligator/core/manifold-base.hpp"

namespace aligator {

template struct DynamicsModelTpl<context::Scalar>;
template struct DynamicsDataTpl<context::Scalar>;

} // namespace aligator
