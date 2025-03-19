#pragma once

#include "aligator/solvers/workspace-base.hpp"
#include "aligator/solvers/value-function.hpp"
#include <Eigen/Cholesky>

namespace aligator {

/// Workspace for solver SolverFDDP.
template <typename Scalar> struct WorkspaceFDDPTpl : WorkspaceBaseTpl<Scalar> {
  using VParams = ValueFunctionTpl<Scalar>;
  using QParams = QFunctionTpl<Scalar>;
  using Base = WorkspaceBaseTpl<Scalar>;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base::nsteps;
  using Base::trial_us;
  using Base::trial_xs;

  /// State increment
  std::vector<VectorXs> dxs;
  /// Control increment
  std::vector<VectorXs> dus;
  /// Storage for the product \f$Q_{uu}k\f$
  std::vector<VectorXs> Quuks_;
  std::vector<VectorXs> ftVxx_;
  /// Buffer for KKT system right-hand sides.
  std::vector<MatrixXs> kktRhs;
  /// LLT struct for each KKT system.
  std::vector<Eigen::LLT<MatrixXs>> llts_;
  /// Value function parameter storage
  std::vector<VParams> value_params;
  /// Q-function storage
  std::vector<QParams> q_params;

  /// Temporary storage for jacobian transpose-Q-hessian product
  using RowMatrixXs = Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>;
  std::vector<RowMatrixXs> JtH_temp_;

  Scalar dg_ = 0.;
  Scalar dq_ = 0.;
  Scalar dv_ = 0.;
  Scalar d1_ = 0.;
  Scalar d2_ = 0.;

  WorkspaceFDDPTpl() : Base(), value_params(), q_params() {}
  explicit WorkspaceFDDPTpl(const TrajOptProblemTpl<Scalar> &problem);

  void cycleLeft();
};

} // namespace aligator

#include "./workspace.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./workspace.txx"
#endif
