#pragma once

#include "aligator/context.hpp"
#include "solver-proxddp.hpp"

namespace aligator {

extern template struct SolverProxDDPTpl<context::Scalar>;

} // namespace aligator
