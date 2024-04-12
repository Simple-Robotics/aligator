#pragma once

#include "riccati-impl.hpp"

#include <aligator/context.hpp>

namespace aligator {
namespace gar {
extern template struct StageFactor<context::Scalar>;
extern template struct ProximalRiccatiKernel<context::Scalar>;
} // namespace gar
} // namespace aligator
