#pragma once

#include "proxddp/context.hpp"
#include "proxddp/modelling/dynamics/multibody-free-fwd.hpp"

namespace aligator {
namespace dynamics {

extern template struct MultibodyFreeFwdDynamicsTpl<context::Scalar>;
extern template struct MultibodyFreeFwdDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
