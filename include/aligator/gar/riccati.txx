#pragma once

#include "./riccati.hpp"
#include "aligator/context.hpp"

namespace aligator {
namespace gar {

extern template class ProximalRiccatiSolver<context::Scalar>;

} // namespace gar
} // namespace aligator
