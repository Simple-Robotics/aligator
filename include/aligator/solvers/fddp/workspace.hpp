#pragma once

#include "proxddp/core/workspace-base.hpp"
#include <Eigen/Cholesky>

namespace aligator {

/// Workspace for solver SolverFDDP.
template <typename Scalar> struct WorkspaceFDDPTpl : WorkspaceBaseTpl<Scalar> {
  using Base = WorkspaceBaseTpl<Scalar>;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base::q_params;
  using Base::trial_us;
  using Base::trial_xs;
  using Base::value_params;

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

  /// Temporary storage for jacobian transpose-Q-hessian product
  using RowMatrixXs = Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>;
  std::vector<RowMatrixXs> JtH_temp_;

  Scalar dg_ = 0.;
  Scalar dq_ = 0.;
  Scalar dv_ = 0.;
  Scalar d1_ = 0.;
  Scalar d2_ = 0.;

  WorkspaceFDDPTpl() : Base() {}
  explicit WorkspaceFDDPTpl(const TrajOptProblemTpl<Scalar> &problem);
  ~WorkspaceFDDPTpl() = default;
  WorkspaceFDDPTpl(const WorkspaceFDDPTpl &) = default;
  WorkspaceFDDPTpl &operator=(const WorkspaceFDDPTpl &) = default;
  WorkspaceFDDPTpl(WorkspaceFDDPTpl &&) = default;
  WorkspaceFDDPTpl &operator=(WorkspaceFDDPTpl &&) = default;

  void cycleLeft() override;
};

} // namespace aligator

#include "./workspace.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./workspace.txx"
#endif
