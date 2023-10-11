#pragma once

#include "./parlqr.hpp"

namespace aligator {
extern template struct LQRFactor<context::Scalar>;
extern template class LQRTreeSolver<context::Scalar>;
} // namespace aligator
