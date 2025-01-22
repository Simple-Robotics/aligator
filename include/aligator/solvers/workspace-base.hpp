/// @file    workspace.hpp
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/fwd.hpp"
#include "solver-util.hpp"
#include "aligator/core/traj-opt-data.hpp"
#include "aligator/utils/mpc-util.hpp"

namespace aligator {

/// Base workspace struct for the algorithms.
template <typename Scalar> struct WorkspaceBaseTpl {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

protected:
  // Whether the workspace was initialized.
  bool m_isInitialized;

public:
  /// Number of steps in the problem.
  std::size_t nsteps;
  /// Problem data.
  TrajOptDataTpl<Scalar> problem_data;

  /// @name Linesearch data
  /// @{
  std::vector<VectorXs> trial_xs;
  std::vector<VectorXs> trial_us;
  /// @}

  /// Dynamical infeasibility gaps
  std::vector<VectorXs> dyn_slacks;

  WorkspaceBaseTpl() : m_isInitialized(false), problem_data() {}

  explicit WorkspaceBaseTpl(const TrajOptProblemTpl<Scalar> &problem);

  ~WorkspaceBaseTpl() = default;

  bool isInitialized() const { return m_isInitialized; }

  /// @brief   Cycle the workspace data to the left.
  /// @details Useful in model-predictive control (MPC) applications.
  void cycleLeft();

  /// @brief Same as cycleLeft(), but add a StageDataTpl to problem_data.
  /// @details The implementation pushes back on top of the vector of
  /// StageDataTpl, rotates left, then pops the first element back out.
  void cycleAppend(shared_ptr<StageDataTpl<Scalar>> data) {
    problem_data.stage_data.emplace_back(data);
    this->cycleLeft();
    problem_data.stage_data.pop_back();
  }
};

/* impl */

template <typename Scalar>
WorkspaceBaseTpl<Scalar>::WorkspaceBaseTpl(
    const TrajOptProblemTpl<Scalar> &problem)
    : m_isInitialized(true), nsteps(problem.numSteps()), problem_data(problem) {
  problem.initializeSolution(trial_xs, trial_us);
}

template <typename Scalar> void WorkspaceBaseTpl<Scalar>::cycleLeft() {
  rotate_vec_left(problem_data.stage_data);

  rotate_vec_left(trial_xs);
  rotate_vec_left(trial_us);

  rotate_vec_left(dyn_slacks, 1);
}

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "workspace-base.txx"
#endif
