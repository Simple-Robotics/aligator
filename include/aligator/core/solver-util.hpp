/// @file solver-util.hpp
/// @brief Common utilities for all solvers.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/traj-opt-problem.hpp"

namespace aligator {

/// @brief Default-intialize a trajectory to the neutral states for each state
/// space at each stage.
template <typename Scalar>
void xs_default_init(const TrajOptProblemTpl<Scalar> &problem,
                     std::vector<typename math_types<Scalar>::VectorXs> &xs) {
  const std::size_t nsteps = problem.numSteps();
  xs.resize(nsteps + 1);
  if (problem.initCondIsStateError()) {
    xs[0] = problem.getInitState();
  } else {
    if (problem.stages_.size() > 0) {
      xs[0] = problem.stages_[0]->xspace().neutral();
    } else {
      ALIGATOR_RUNTIME_ERROR(
          "The problem should have either a StateErrorResidual as an initial "
          "condition or at least one stage.");
    }
  }
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModelTpl<Scalar> &sm = *problem.stages_[i];
    xs[i + 1] = sm.xspace_next().neutral();
  }
}

/// @brief Default-initialize a controls trajectory from the neutral element of
/// each control space.
template <typename Scalar>
void us_default_init(const TrajOptProblemTpl<Scalar> &problem,
                     std::vector<typename math_types<Scalar>::VectorXs> &us) {
  const std::size_t nsteps = problem.numSteps();
  us.resize(nsteps);
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModelTpl<Scalar> &sm = *problem.stages_[i];
    us[i] = sm.uspace().neutral();
  }
}

template <typename Scalar>
auto problemInitializeSolution(const TrajOptProblemTpl<Scalar> &problem) {
  using VectorXs = typename math_types<Scalar>::VectorXs;
  std::vector<VectorXs> xs, us, vs, lbdas;
  const size_t nsteps = problem.numSteps();
  xs_default_init(problem, xs);
  us_default_init(problem, us);
  // initialize multipliers...
  vs.resize(nsteps + 1);
  lbdas.resize(nsteps + 1);
  lbdas[0].setZero(problem.init_condition_->nr);
  for (size_t i = 0; i < nsteps; i++) {
    const StageModelTpl<Scalar> &sm = *problem.stages_[i];
    lbdas[i + 1].setZero(sm.ndx2());
    vs[i].setZero(sm.nc());
  }

  if (!problem.term_cstrs_.empty()) {
    vs[nsteps].setZero(problem.term_cstrs_.totalDim());
  }

  return std::make_tuple(std::move(xs), std::move(us), std::move(vs),
                         std::move(lbdas));
}

/// @brief Check the input state-control trajectory is a consistent warm-start
/// for the output.
template <typename Scalar>
void check_trajectory_and_assign(
    const TrajOptProblemTpl<Scalar> &problem,
    const typename math_types<Scalar>::VectorOfVectors &xs_init,
    const typename math_types<Scalar>::VectorOfVectors &us_init,
    typename math_types<Scalar>::VectorOfVectors &xs_out,
    typename math_types<Scalar>::VectorOfVectors &us_out) {
  const std::size_t nsteps = problem.numSteps();
  xs_out.reserve(nsteps + 1);
  us_out.reserve(nsteps);
  if (xs_init.size() == 0) {
    xs_default_init(problem, xs_out);
  } else {
    if (xs_init.size() != (nsteps + 1)) {
      ALIGATOR_RUNTIME_ERROR("warm-start for xs has wrong size!");
    }
    xs_out = xs_init;
  }
  if (us_init.size() == 0) {
    us_default_init(problem, us_out);
  } else {
    if (us_init.size() != nsteps) {
      ALIGATOR_RUNTIME_ERROR("warm-start for us has wrong size!");
    }
    us_out = us_init;
  }
}

} // namespace aligator
