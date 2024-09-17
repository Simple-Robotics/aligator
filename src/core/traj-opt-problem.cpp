/// @file
/// @copyright Copyright (C) LAAS-CNRS, INRIA 2024
#include "aligator/core/traj-opt-problem.hxx" // actual implementation file
#include "aligator/core/traj-opt-data.hpp"
#include "aligator/core/cost-abstract.hpp"

namespace aligator {

template struct TrajOptProblemTpl<context::Scalar>;

} // namespace aligator
