#include "aligator/gar/lqr-knot.hpp"

namespace aligator {
namespace gar {
template struct LQRKnotTpl<context::Scalar>;
template struct LQRProblemTpl<context::Scalar>;
template auto
lqrDenseMatrix<context::Scalar>(const LQRProblemTpl<context::Scalar> &,
                                context::Scalar, context::Scalar);
} // namespace gar

} // namespace aligator
