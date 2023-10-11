#include "aligator/gar/lqr-knot.hpp"

namespace aligator {
namespace gar {
template struct LQRKnot<context::Scalar>;
template struct LQRProblem<context::Scalar>;
template auto
lqrDenseMatrix<context::Scalar>(const std::vector<LQRKnot<context::Scalar>> &,
                                context::Scalar, context::Scalar);
} // namespace gar

} // namespace aligator
