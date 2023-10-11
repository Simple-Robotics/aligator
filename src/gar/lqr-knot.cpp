#include "aligator/gar/lqr-knot.hpp"

namespace aligator {
namespace gar {
template struct LQRKnot<context::Scalar>;
template struct LQRProblem<context::Scalar>;
} // namespace gar

} // namespace aligator
