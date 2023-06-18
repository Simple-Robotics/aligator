#pragma once

#include "./multibody-constraint-fwd.hpp"

namespace proxddp {
namespace dynamics {

extern template struct MultibodyConstraintFwdDynamicsTpl<context::Scalar>;
extern template struct MultibodyConstraintFwdDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace proxddp
