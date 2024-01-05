#pragma once

#include "./integrator-explicit.hpp"

namespace aligator {
namespace dynamics {

extern template struct ExplicitIntegratorAbstractTpl<context::Scalar>;
extern template struct ExplicitIntegratorDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
