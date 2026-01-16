/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2026 INRIA
/// @brief Implementation file. Include when template definitions required.
#pragma once

#include "results.hpp"

#include "aligator/tracy.hpp"
#include "aligator/core/traj-opt-problem.hpp"
#include "aligator/utils/mpc-util.hpp"

namespace aligator {

template <typename Scalar>
ResultsTpl<Scalar>::ResultsTpl(const TrajOptProblemTpl<Scalar> &problem)
    : Base() {
  if (!problem.checkIntegrity())
    ALIGATOR_RUNTIME_ERROR("Problem failed integrity check.");

  const std::size_t nsteps = problem.numSteps();
  problem.initializeSolution(xs, us, vs, lams);

  gains_.resize(nsteps + 1);
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModelTpl<Scalar> &sm = *problem.stages_[i];
    const int nu = sm.nu();
    const int nc = sm.nc();
    const int ndx2 = sm.ndx2();
    const int ndx = sm.ndx1();
    gains_[i].setZero(nu + nc + ndx2, ndx + 1);
  }

  // terminal constraints
  {
    const int ndx = internal::problem_last_ndx_helper(problem);
    const long nc = problem.term_cstrs_.totalDim();
    gains_[nsteps].setZero(nc, ndx + 1);
  }

  this->m_isInitialized = true;
}

template <typename Scalar>
void ResultsTpl<Scalar>::cycleAppend(const TrajOptProblemTpl<Scalar> &problem,
                                     const ConstVectorRef &x0) {
  ALIGATOR_TRACY_ZONE_SCOPED;
  xs.front() = xs.back();
  us.front() = us.back();

  rotate_vec_left(xs);
  rotate_vec_left(us);
  rotate_vec_left(vs, 0, 1);
  rotate_vec_left(lams);
  rotate_vec_left(gains_, 0, 1);

  xs.front() = x0;

  const std::size_t nsteps = problem.numSteps();
  const StageModelTpl<Scalar> &sm = *problem.stages_[nsteps - 1];
  gains_[nsteps - 1].setZero(sm.nu() + sm.nc() + sm.ndx2(), sm.ndx1() + 1);
  vs[nsteps - 1].setZero(sm.nc());
  lams[nsteps].setZero(sm.ndx2());

  lams[0].setZero(problem.init_constraint_->nr);

  if (!problem.term_cstrs_.empty()) {
    const int ndx_ter = internal::problem_last_ndx_helper(problem);
    const long nc = problem.term_cstrs_.totalDim();
    gains_[nsteps].setZero(nc, ndx_ter + 1);
    vs[nsteps].setZero(nc);
  }
}
} // namespace aligator
