#pragma once

#include "aligator/context.hpp"
#include "aligator/core/traj-opt-problem.hpp"

namespace aligator {

extern template struct TrajOptProblemTpl<context::Scalar>;

} // namespace aligator
