/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "./results.hpp"
#include "aligator/core/solver-util.hpp"

#include <fmt/format.h>

namespace aligator {

template <typename Scalar>
ResultsTpl<Scalar>::ResultsTpl(const TrajOptProblemTpl<Scalar> &problem)
    : Base() {

  const std::size_t nsteps = problem.numSteps();
  std::tie(xs, us, vs, lams) = problemInitializeSolution(problem);

  assert(xs.size() == nsteps + 1);
  assert(us.size() == nsteps);

  this->m_isInitialized = true;
}
} // namespace aligator
