#pragma once

#include "./riccati.hpp"
#include "aligator/context.hpp"

namespace aligator {
namespace gar {

extern template auto
lqrDenseMatrix<context::Scalar>(const std::vector<LQRKnot<context::Scalar>> &,
                                context::Scalar, context::Scalar);
extern template class ProximalRiccatiSolver<context::Scalar>;

} // namespace gar
} // namespace aligator
