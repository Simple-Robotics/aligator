#pragma once

#include "aligator/context.hpp"
#include "aligator/gar/lqr-knot.hpp"

namespace aligator {
namespace gar {
extern template struct LQRKnotTpl<context::Scalar>;
extern template struct LQRProblemTpl<context::Scalar>;
extern template auto
lqrDenseMatrix<context::Scalar>(const LQRProblemTpl<context::Scalar> &,
                                context::Scalar, context::Scalar);
} // namespace gar
} // namespace aligator
