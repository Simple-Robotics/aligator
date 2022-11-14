#pragma once

#include "proxddp/core/workspace.hpp"

namespace proxddp {

/// Workspace for solver SolverFDDP.
template <typename Scalar> struct WorkspaceFDDPTpl : WorkspaceBaseTpl<Scalar> {
  using Base = WorkspaceBaseTpl<Scalar>;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base::nsteps;
  using Base::q_params;
  using Base::trial_us;
  using Base::trial_xs;
  using Base::value_params;

  /// Value of `f(x_i, u_i)`
  std::vector<VectorXs> xnexts_;
  /// State increment
  std::vector<VectorXs> dxs;
  /// Control increment
  std::vector<VectorXs> dus;
  std::vector<VectorXs> Quuks_;
  std::vector<VectorXs> ftVxx_;
  /// Buffer for KKT matrices.
  std::vector<MatrixXs> kkt_mat_bufs;
  /// Buffer for KKT system right-hand sides.
  std::vector<MatrixXs> kkt_rhs_bufs;
  /// LLT struct for each KKT system.
  std::vector<Eigen::LLT<MatrixXs>> llts_;

  Scalar dg_ = 0.;
  Scalar dq_ = 0.;
  Scalar dv_ = 0.;

  explicit WorkspaceFDDPTpl(const TrajOptProblemTpl<Scalar> &problem);

  void cycle_left();

  /// @brief Same as cycle_left(), but add a StageDataTpl to problem_data.
  /// @details The implementation pushes back on top of the vector of
  /// StageDataTpl, rotates left, then pops the first element back out.
  void cycle_append(const shared_ptr<StageModelTpl<Scalar>> &stage) {
    auto sd = stage->createData();
    this->problem_data.stage_data.push_back(sd);
    this->cycle_left();
    this->problem_data.stage_data.pop_back();
  }
};

} // namespace proxddp

#include "proxddp/fddp/workspace.hxx"
