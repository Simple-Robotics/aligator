#include "aligator/gar/lqr-knot.hpp"

namespace aligator {
namespace gar {
template struct LQRKnot<context::Scalar>;
template struct LQRProblem<context::Scalar>;
template auto
lqrDenseMatrix<context::Scalar>(const LQRProblem<context::Scalar> &,
                                context::Scalar, context::Scalar);
} // namespace gar

} // namespace aligator
