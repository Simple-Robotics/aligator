#include "aligator/gar/riccati.hpp"

namespace aligator::gar {

template class ProximalRiccatiSolverBackward<context::Scalar>;
template class ProximalRiccatiSolverForward<context::Scalar>;

} // namespace aligator::gar
