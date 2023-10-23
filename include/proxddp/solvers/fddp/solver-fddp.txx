#pragma once

#include "proxddp/context.hpp"

#include "./solver-fddp.hpp"

namespace proxddp {

extern template struct SolverFDDP<context::Scalar>;

}
