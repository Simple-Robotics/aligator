#pragma once

#include "aligator/context.hpp"
#include "aligator/core/traj-opt-data.hpp"

namespace aligator {

extern template struct TrajOptDataTpl<context::Scalar>;
extern template context::Scalar computeTrajectoryCost<context::Scalar>(
    const context::TrajOptData &problem_data);

} // namespace aligator
