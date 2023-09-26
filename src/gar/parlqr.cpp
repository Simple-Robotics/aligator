#include "aligator/gar/parlqr.hpp"

namespace aligator {

template struct LQRFactor<context::Scalar>;
template class LQRTreeSolver<context::Scalar>;

} // namespace aligator
