/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/dynamics/centroidal-fwd.hpp"

namespace aligator {
namespace dynamics {

extern template struct CentroidalFwdDynamicsTpl<context::Scalar>;
extern template struct CentroidalFwdDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
