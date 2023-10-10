#pragma once

#include "./riccati.hpp"
#include "aligator/context.hpp"

namespace aligator {
namespace gar {

extern template class ProximalRiccatiSolverBackward<context::Scalar>;
extern template class ProximalRiccatiSolverForward<context::Scalar>;

} // namespace gar
} // namespace aligator
