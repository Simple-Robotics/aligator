/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#include "aligator/modelling/dynamics/multibody-free-fwd.hpp"

namespace aligator {
namespace dynamics {

template struct MultibodyFreeFwdDynamicsTpl<context::Scalar>;
template struct MultibodyFreeFwdDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
