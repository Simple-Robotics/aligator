#pragma once

#include "aligator/context.hpp"
#include "aligator/gar/lqr-knot.hpp"

namespace aligator {
namespace gar {

extern template struct LQRKnot<context::Scalar>;
extern template struct LQRProblem<context::Scalar>;
extern template auto
lqrDenseMatrix<context::Scalar>(const std::vector<LQRKnot<context::Scalar>> &,
                                context::Scalar, context::Scalar);
} // namespace gar
} // namespace aligator
