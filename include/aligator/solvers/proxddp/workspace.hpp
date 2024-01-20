/// @file    workspace.hpp
/// @brief   Define workspace for the ProxDDP solver.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/workspace-base.hpp"
#include "aligator/core/alm-weights.hpp"
#include "aligator/gar/lqr-problem.hpp"
#include "aligator/core/constraint-set-product.hpp"

#include <array>

namespace aligator {

template <typename Scalar>
auto getConstraintProductSet(const ConstraintStackTpl<Scalar> &constraints) {
  ConstraintSetProductTpl<Scalar> out;
  for (size_t i = 0; i < constraints.size(); i++) {
    out.components.push_back(constraints[i].set.get());
  }
  out.nrs = constraints.dims();
  return out;
}

/// @brief Workspace for solver SolverProxDDP.
///
/// @details This struct holds data for the Riccati forward and backward passes,
///          the primal-dual steps, problem data...
template <typename Scalar> struct WorkspaceTpl : WorkspaceBaseTpl<Scalar> {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using StageModel = StageModelTpl<Scalar>;
  using Base = WorkspaceBaseTpl<Scalar>;
  using VecBool = Eigen::Matrix<bool, Eigen::Dynamic, 1>;
  using CstrProxScaler = ConstraintProximalScalerTpl<Scalar>;
  using KnotType = gar::LQRKnotTpl<Scalar>;
  using ConstraintSetProduct = ConstraintSetProductTpl<Scalar>;
  using BlkJacobianType = BlkMatrix<MatrixXs, -1, 2>; // jacobians

  using Base::dyn_slacks;
  using Base::nsteps;
  using Base::problem_data;

  gar::LQRProblemTpl<Scalar> lqr_problem;   //< Linear-quadratic subproblem
  std::vector<CstrProxScaler> cstr_scalers; //< Scaling for the constraints

  /// @name Lagrangian Gradients
  /// @{
  std::vector<VectorXs> Lxs_; //< State gradients
  std::vector<VectorXs> Lus_; //< Control gradients
  std::vector<VectorXs> Lvs_; //< Path multiplier gradients
  std::vector<VectorXs> Lds_; //< Costate gradients
  /// @}

  /// @name Trial primal-dual step
  /// @{
  using Base::trial_us;
  using Base::trial_xs;
  std::vector<VectorXs> trial_vs;
  std::vector<VectorXs> trial_lams;
  /// @}

  /// @name Lagrange multipliers.
  /// @{
  std::vector<VectorXs> lams_plus;
  std::vector<VectorXs> lams_pdal;
  std::vector<VectorXs> vs_plus;
  std::vector<VectorXs> vs_pdal;
  /// @}

  /// Shifted constraints the projection operators should be applied to.
  std::vector<VectorXs> shifted_constraints;
  /// Projected path constraint Jacobians (used to symmetrize the LQ subproblem)
  std::vector<BlkJacobianType> constraintProjJacobians;
  /// Masks for active constraint sets
  std::vector<VecBool> active_constraints;
  /// Cartesian products of the constraint sets of each stage.
  std::vector<ConstraintSetProduct> constraintProductOperators;

  /// @name Primal-dual steps
  /// @{
  std::vector<VectorXs> dxs;
  std::vector<VectorXs> dus;
  std::vector<VectorXs> dvs;
  std::vector<VectorXs> dlams;
  /// @}

  /// @name Previous external/proximal iterates
  /// @{
  std::vector<VectorXs> prev_xs;
  std::vector<VectorXs> prev_us;
  std::vector<VectorXs> prev_vs;
  std::vector<VectorXs> prev_lams;
  /// @}

  /// Subproblem termination criterion for each stage.
  VectorXs stage_inner_crits;
  /// Constraint violation measures for each stage and constraint.
  VectorXs stage_cstr_violations;
  /// Stagewise infeasibilities
  std::vector<VectorXs> stage_infeasibilities;
  /// Dual infeasibility in the states for each stage of the problem.
  VectorXs state_dual_infeas;
  /// Dual infeasibility in the controls for each stage of the problem.
  VectorXs control_dual_infeas;
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
