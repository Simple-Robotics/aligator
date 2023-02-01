#pragma once

#include "proxddp/context.hpp"

#include "proxddp/fddp/solver-fddp.hpp"

namespace proxddp {

extern template struct SolverFDDP<context::Scalar>;

}
