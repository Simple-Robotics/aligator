#pragma once

#include "proxddp/context.hpp"

#include "proxddp/fddp/solver-fddp.hpp"

namespace proxddp {

// instantiate constructor

extern template
SolverFDDP<context::Scalar>::SolverFDDP(const context::Scalar, VerboseLevel, const context::Scalar, const std::size_t);


}
