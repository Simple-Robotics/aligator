/// @file solver-util.hpp
/// @brief Common utilities for all solvers.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/traj-opt-problem.hpp"
#include "proxddp/utils/exceptions.hpp"

namespace proxddp {

/// @brief Default-intialize a trajectory to the neutral states for each state
/// space at each stage.
template <typename Scalar>
void xs_default_init(const TrajOptProblemTpl<Scalar> &problem,
                     std::vector<typename math_types<Scalar>::VectorXs> &xs) {
  using Manifold = ManifoldAbstractTpl<Scalar>;
  const std::size_t nsteps = problem.numSteps();
  xs.resize(nsteps + 1);
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModelTpl<Scalar> &sm = *problem.stages_[i];
    xs[i] = sm.xspace().neutral();
  }
  const Manifold &space = problem.stages_.back()->xspace_next();
  xs[nsteps] = space.neutral();
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
      PROXDDP_RUNTIME_ERROR("warm-start for xs has wrong size!");
    }
    xs_out = xs_init;
  }
  if (us_init.size() == 0) {
    us_default_init(problem, us_out);
  } else {
    if (us_init.size() != nsteps) {
      PROXDDP_RUNTIME_ERROR("warm-start for us has wrong size!");
    }
    us_out = us_init;
  }
}

} // namespace proxddp

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxddp/core/solver-util.txx"
#endif
