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
  problem.checkIntegrity();

  const std::size_t nsteps = problem.numSteps();
  std::tie(xs, us, vs, lams) = problemInitializeSolution(problem);

  assert(xs.size() == nsteps + 1);
  assert(us.size() == nsteps);

  gains_.resize(nsteps + 1);
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModelTpl<Scalar> &sm = *problem.stages_[i];
    const int np = sm.numPrimal();
    const int nd = sm.numDual();
    const int ndx = sm.ndx1();
    gains_[i].setZero(np + nd, ndx + 1);
  }

  // terminal constraints
  {
    const int ndx = internal::problem_last_ndx_helper(problem);
    const long nc = problem.term_cstrs_.totalDim();
    gains_[nsteps].setZero(nc, ndx + 1);
  }

  this->m_isInitialized = true;
}
} // namespace aligator
