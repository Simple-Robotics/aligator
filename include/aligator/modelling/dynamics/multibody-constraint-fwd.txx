/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "./multibody-constraint-fwd.hpp"

namespace aligator {
namespace dynamics {

extern template struct MultibodyConstraintFwdDynamicsTpl<context::Scalar>;
extern template struct MultibodyConstraintFwdDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
