/// @file    workspace.hpp
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/value-function.hpp"
#include "proxddp/core/solver-util.hpp"

namespace proxddp {

/// Base workspace struct for the algorithms.
template <typename Scalar> struct WorkspaceBaseTpl {
  using VParams = value_function<Scalar>;
  using QParams = q_function<Scalar>;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  /// Number of steps in the problem.
  const std::size_t nsteps;
  /// Problem data.
  TrajOptDataTpl<Scalar> problem_data;

  /// @name Linesearch data
  /// @{
  std::vector<VectorXs> trial_xs;
  std::vector<VectorXs> trial_us;
  /// @}

  /// Feasibility gaps
  std::vector<VectorXs> dyn_slacks;
  /// Value function parameter storage
  std::vector<VParams> value_params;
  /// Q-function storage
  std::vector<QParams> q_params;

  explicit WorkspaceBaseTpl(const TrajOptProblemTpl<Scalar> &problem);

  /// @brief   Cycle the workspace data to the left.
  /// @details Useful in model-predictive control (MPC) applications.
  void cycle_left();
};

/* impl */

template <typename Scalar>
WorkspaceBaseTpl<Scalar>::WorkspaceBaseTpl(
    const TrajOptProblemTpl<Scalar> &problem)
    : nsteps(problem.numSteps()), problem_data(problem) {
  trial_xs.resize(nsteps + 1);
  trial_us.resize(nsteps);
  xs_default_init(problem, trial_xs);
  us_default_init(problem, trial_us);
}

template <typename Scalar> void WorkspaceBaseTpl<Scalar>::cycle_left() {
  rotate_vec_left(problem_data.stage_data);

  rotate_vec_left(trial_xs);
  rotate_vec_left(trial_us);

  rotate_vec_left(dyn_slacks, 1);

  rotate_vec_left(value_params);
  rotate_vec_left(q_params);
}

} // namespace proxddp
