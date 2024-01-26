#pragma once

#include "aligator/context.hpp"

#include "./solver-fddp.hpp"

namespace aligator {

extern template struct SolverFDDPTpl<context::Scalar>;

}
