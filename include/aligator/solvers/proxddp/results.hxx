/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
/// @brief Implementation file. Include when template definitions required.
#pragma once

#include "./results.hpp"
#include "aligator/core/solver-util.hpp"
#include "aligator/utils/mpc-util.hpp"

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

template <typename Scalar>
void ResultsTpl<Scalar>::cycleAppend(const TrajOptProblemTpl<Scalar> &problem,
                                     const Eigen::VectorXd &x0) {
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
  gains_[nsteps - 1].setZero(sm.numPrimal() + sm.numDual(), sm.ndx1() + 1);
  vs[nsteps - 1].setZero(sm.nc());
  lams[nsteps].setZero(sm.ndx2());

  lams[0].setZero(problem.init_condition_->nr);

  if (!problem.term_cstrs_.empty()) {
    const int ndx_ter = internal::problem_last_ndx_helper(problem);
    const long nc = problem.term_cstrs_.totalDim();
    gains_[nsteps].setZero(nc, ndx_ter + 1);
    vs[nsteps].setZero(nc);
  }
}
} // namespace aligator
