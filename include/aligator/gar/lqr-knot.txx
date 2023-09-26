#pragma once

#include "aligator/context.hpp"
#include "aligator/gar/lqr-knot.hpp"

namespace aligator {

extern template struct LQRKnot<context::Scalar>;
extern template struct LQRProblem<context::Scalar>;

} // namespace aligator
