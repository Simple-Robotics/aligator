/// @file    workspace.hpp
/// @brief   Define workspace for the ProxDDP solver.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/workspace-base.hpp"
#include "proxddp/core/proximal-penalty.hpp"
#include "proxddp/core/alm-weights.hpp"

#include <array>

#include <proxnlp/ldlt-allocator.hpp>

namespace proxddp {

using proxnlp::LDLTChoice;

/** @brief Workspace for solver SolverProxDDP.
 *
 * @details This struct holds data for the Riccati forward and backward passes,
 *          the primal-dual steps, problem data...
 */
template <typename Scalar> struct WorkspaceTpl : WorkspaceBaseTpl<Scalar> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using ProxPenalty = ProximalPenaltyTpl<Scalar>;
  using ProxData = typename ProxPenalty::Data;
  using StageModel = StageModelTpl<Scalar>;
  using Base = WorkspaceBaseTpl<Scalar>;
  using VecBool = Eigen::Matrix<bool, Eigen::Dynamic, 1>;
  using CstrProxScaler = ConstraintProximalScalerTpl<Scalar>;

  using Base::dyn_slacks;
  using Base::nsteps;
  using Base::problem_data;
  using Base::q_params;
  using Base::trial_us;
  using Base::trial_xs;
  using Base::value_params;

  /// Proximal penalty data.
  std::vector<shared_ptr<ProxData>> prox_datas;

  /// Proximal algo scalers for the constraints
  std::vector<CstrProxScaler> cstr_scalers;

  /// @name Lagrangian Gradients
  /// @{
  std::vector<VectorXs> Lxs_;
  std::vector<VectorXs> Lus_;
  /// @}

  /// Lagrange multipliers for ALM & linesearch.
  std::vector<VectorXs> trial_lams;
  std::vector<VectorXs> lams_plus;
  std::vector<VectorXs> lams_pdal;
  /// Shifted constraints the projection operators should be applied to.
  std::vector<VectorXs> shifted_constraints;
  std::vector<MatrixXs> proj_jacobians;
  std::vector<VecBool> active_constraints;

  /// @name Riccati gains, memory buffers for primal-dual steps
  /// @{
  std::vector<VectorXs> pd_step_;
  std::vector<VectorRef> dxs;
  std::vector<VectorRef> dus;
  std::vector<VectorRef> dlams;

  /// Buffer for KKT matrix
  std::vector<MatrixXs> kkt_mats_;
  /// Buffer for KKT right hand side
  std::vector<MatrixXs> kkt_rhs_;
  /// Linear system residual buffers: used for iterative refinement
  std::vector<MatrixXs> kkt_resdls_;

  using LDLT = proxnlp::linalg::ldlt_base<Scalar>;
  /// LDLT solvers
  std::vector<unique_ptr<LDLT>> ldlts_;

  /// @}

  /// @name Previous external/proximal iterates
  /// @{

  std::vector<VectorXs> prev_xs;
  std::vector<VectorXs> prev_us;
  std::vector<VectorXs> lams_prev;

  /// @}

  /// Subproblem termination criterion for each stage.
  VectorXs stage_inner_crits;
  /// Constraint violation for each stage and each constraint of the
  /// TrajOptProblemTpl.
  std::vector<VectorXs> stage_prim_infeas;
  /// Dual infeasibility for each stage of the TrajOptProblemTpl.
  VectorXs stage_dual_infeas;

  /// Overall subproblem termination criterion.
  Scalar inner_criterion = 0.;

  WorkspaceTpl() : Base() {}
  WorkspaceTpl(const TrajOptProblemTpl<Scalar> &problem,
               LDLTChoice ldlt_choice = LDLTChoice::DENSE);
  WorkspaceTpl(const WorkspaceTpl &) = default;
  WorkspaceTpl(WorkspaceTpl &&) = default;
  ~WorkspaceTpl() = default;

  WorkspaceTpl &operator=(const WorkspaceTpl &) = default;
  WorkspaceTpl &operator=(WorkspaceTpl &&) = default;

  void cycleLeft() override;

  template <typename T>
  friend std::ostream &operator<<(std::ostream &oss,
                                  const WorkspaceTpl<T> &self);

  void configureScalers(const TrajOptProblemTpl<Scalar> &problem,
                        const Scalar &mu);
};

namespace {
using proxnlp::linalg::isize;
using proxnlp::linalg::ldlt_base;
} // namespace

// fwd declaration

/// Allocate an LDLT solver, perform no analysis.
template <typename Scalar>
unique_ptr<ldlt_base<Scalar>>
allocate_ldlt_algorithm(const std::vector<isize> &nprims,
                        const std::vector<isize> &nduals, LDLTChoice choice);

} // namespace proxddp

template <typename Scalar>
struct fmt::formatter<proxddp::WorkspaceTpl<Scalar>> : fmt::ostream_formatter {
};

#include "proxddp/core/workspace.hxx"

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxddp/core/workspace.txx"
#endif
