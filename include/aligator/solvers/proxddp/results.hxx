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
  xs_default_init(problem, xs);
  us_default_init(problem, us);
  lams.resize(nsteps + 2);
  vs.resize(nsteps + 1);

  lams[0].setZero(problem.init_condition_->nr);

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModelTpl<Scalar> &stage = *problem.stages_[i];
    lams[i + 1].setZero(stage.ndx2());
    vs[i].setZero(stage.nc());
  }

  if (!problem.term_cstrs_.empty()) {
    const long nc = problem.term_cstrs_.totalDim();
    vs[nsteps].setZero(nc);
  }
  assert(xs.size() == nsteps + 1);
  assert(us.size() == nsteps);

  this->m_isInitialized = true;
}
} // namespace aligator
