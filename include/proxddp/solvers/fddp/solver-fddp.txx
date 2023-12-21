#pragma once

#include "proxddp/context.hpp"

#include "./solver-fddp.hpp"

namespace aligator {

extern template struct SolverFDDP<context::Scalar>;

}
