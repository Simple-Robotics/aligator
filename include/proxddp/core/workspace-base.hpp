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

  /// Feasibility gaps
  std::vector<VectorXs> dyn_slacks;
  /// Value function parameter storage
  std::vector<VParams> value_params;
  /// Q-function storage
  std::vector<QParams> q_params;

  WorkspaceBaseTpl() : m_isInitialized(false), problem_data() {}

  explicit WorkspaceBaseTpl(const TrajOptProblemTpl<Scalar> &problem);

  virtual ~WorkspaceBaseTpl() = 0;

  bool isInitialized() const { return m_isInitialized; }

  /// @brief   Cycle the workspace data to the left.
  /// @details Useful in model-predictive control (MPC) applications.
  virtual void cycleLeft();

  /// @brief Same as cycleLeft(), but add a StageDataTpl to problem_data.
  /// @details The implementation pushes back on top of the vector of
  /// StageDataTpl, rotates left, then pops the first element back out.
  void cycleAppend(shared_ptr<StageDataTpl<Scalar>> data) {
    problem_data.stage_data.emplace_back(data);
    this->cycleLeft();
    problem_data.stage_data.pop_back();
  }
};

template <typename Scalar> WorkspaceBaseTpl<Scalar>::~WorkspaceBaseTpl() {}

/* impl */

template <typename Scalar>
WorkspaceBaseTpl<Scalar>::WorkspaceBaseTpl(
    const TrajOptProblemTpl<Scalar> &problem)
    : m_isInitialized(true), nsteps(problem.numSteps()), problem_data(problem) {
  trial_xs.resize(nsteps + 1);
  trial_us.resize(nsteps);
  xs_default_init(problem, trial_xs);
  us_default_init(problem, trial_us);
}

template <typename Scalar> void WorkspaceBaseTpl<Scalar>::cycleLeft() {
  rotate_vec_left(problem_data.stage_data);

  rotate_vec_left(trial_xs);
  rotate_vec_left(trial_us);

  rotate_vec_left(dyn_slacks, 1);

  rotate_vec_left(value_params);
  rotate_vec_left(q_params);
}

} // namespace proxddp
