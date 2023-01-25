#pragma once

#include "proxddp/context.hpp"

namespace proxddp {

extern template
SolverProxDDP<context::Scalar>::SolverProxDDP(const context::Scalar, const context::Scalar, const context::Scalar, const std::size_t, VerboseLevel, HessianApprox);

} // namespace proxddp
