#pragma once

#include "proxddp/context.hpp"
#include "proxddp/core/traj-opt-problem.hpp"

namespace aligator {

extern template struct TrajOptProblemTpl<context::Scalar>;

extern template struct TrajOptDataTpl<context::Scalar>;

} // namespace aligator
