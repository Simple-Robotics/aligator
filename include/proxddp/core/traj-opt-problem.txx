#pragma once

#include "proxddp/context.hpp"
#include "proxddp/core/traj-opt-problem.hpp"

namespace proxddp {

namespace {
using StateErrorResidual = StateErrorResidualTpl<context::Scalar>;
}

extern template struct TrajOptProblemTpl<context::Scalar>;

} // namespace proxddp
