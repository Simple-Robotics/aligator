#include "aligator/context.hpp"
#include "aligator/gar/dense-kernel.hpp"

namespace aligator::gar {
template struct DenseKernel<context::Scalar>;
} // namespace aligator::gar
