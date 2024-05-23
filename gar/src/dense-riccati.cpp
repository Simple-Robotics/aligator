#include "aligator/context.hpp"
#include "aligator/gar/dense-riccati.hpp"

namespace aligator::gar {
template class RiccatiSolverDense<context::Scalar>;
} // namespace aligator::gar
