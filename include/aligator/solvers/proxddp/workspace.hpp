/// @file    workspace.hpp
/// @brief   Define workspace for the ProxDDP solver.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/workspace-base.hpp"
#include "aligator/core/proximal-penalty.hpp"
#include "aligator/core/alm-weights.hpp"
#include "aligator/gar/riccati.hpp"

#include <array>
#include <proxsuite-nlp/ldlt-allocator.hpp>

namespace aligator {

using proxsuite::nlp::LDLTChoice;

/** @brief Workspace for solver SolverProxDDP.
 *
 * @details This struct holds data for the Riccati forward and backward passes,
 *          the primal-dual steps, problem data...
 */
template <typename Scalar> struct WorkspaceTpl : WorkspaceBaseTpl<Scalar> {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using StageModel = StageModelTpl<Scalar>;
  using Base = WorkspaceBaseTpl<Scalar>;
  using VecBool = Eigen::Matrix<bool, Eigen::Dynamic, 1>;
  using CstrProxScaler = ConstraintProximalScalerTpl<Scalar>;
  using BlkVec = BlkMatrix<VectorXs, -1>;
  using BlkMat = BlkMatrix<MatrixXs, -1, -1>;
  using ProxRiccati = gar::ProximalRiccatiSolver<Scalar>;

  using Base::dyn_slacks;
  using Base::nsteps;
  using Base::problem_data;
  using Base::q_params;
  using Base::trial_us;
  using Base::trial_xs;
  using Base::value_params;

  gar::LQRProblemTpl<Scalar> lqrData;
  // unique_ptr<ProxRiccati> lqrSolver;

  /// Proximal algo scalers for the constraints
  std::vector<CstrProxScaler> cstr_scalers;

  /// @name Lagrangian Gradients
  /// @{
  std::vector<VectorXs> Lxs_;
  std::vector<VectorXs> Lus_;
  std::vector<VectorXs> Lds_;
  /// @}

  /// Lagrange multipliers for ALM & linesearch.
  std::vector<VectorXs> trial_lams;
  std::vector<VectorXs> lams_plus;
  std::vector<VectorXs> lams_pdal;
  /// Shifted constraints the projection operators should be applied to.
  std::vector<VectorXs> shifted_constraints;
  /// Projected Jacobians, used to symmetrize LQR subproblem
  std::vector<MatrixXs> proj_jacobians;
  std::vector<BlkMat> proj_jacobians_2;
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

  using LDLTVariant = proxsuite::nlp::LDLTVariant<Scalar>;
  /// LDLT solvers
  std::vector<LDLTVariant> ldlts_;

  /// @}

  /// @name Previous external/proximal iterates
  /// @{

  std::vector<VectorXs> prev_xs;
  std::vector<VectorXs> prev_us;
  std::vector<VectorXs> prev_lams;

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
  WorkspaceTpl(const TrajOptProblemTpl<Scalar> &problem);

  void cycleLeft() override;

  template <typename T>
  friend std::ostream &operator<<(std::ostream &oss,
                                  const WorkspaceTpl<T> &self);

  template <typename F>
  void configureScalers(const TrajOptProblemTpl<Scalar> &problem,
                        const Scalar &mu, F &&strat) {
    cstr_scalers.reserve(nsteps + 1);

    for (std::size_t t = 0; t < nsteps; t++) {
      const StageModel &stage = *problem.stages_[t];
      cstr_scalers.emplace_back(stage.constraints_, mu);
      std::forward<F>(strat)(cstr_scalers[t]);
    }

    const ConstraintStackTpl<Scalar> &term_stack = problem.term_cstrs_;
    if (!term_stack.empty()) {
      cstr_scalers.emplace_back(term_stack, mu);
    }
  }
};

} // namespace aligator

template <typename Scalar>
struct fmt::formatter<aligator::WorkspaceTpl<Scalar>> : fmt::ostream_formatter {
};

#include "./workspace.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./workspace.txx"
#endif
