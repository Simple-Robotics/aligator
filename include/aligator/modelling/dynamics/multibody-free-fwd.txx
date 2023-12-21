#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/dynamics/multibody-free-fwd.hpp"

namespace aligator {
namespace dynamics {

extern template struct MultibodyFreeFwdDynamicsTpl<context::Scalar>;
extern template struct MultibodyFreeFwdDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
