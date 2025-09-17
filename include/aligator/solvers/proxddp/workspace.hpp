/// @file    workspace.hpp
/// @brief   Define workspace for the ProxDDP solver.
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/solvers/workspace-base.hpp"
#include "aligator/gar/blk-matrix.hpp"
#include "aligator/gar/lqr-problem.hpp"

#include "aligator/modelling/constraints/constraint-set-product.hpp"

namespace aligator {

template <typename Scalar>
auto getConstraintProductSet(const ConstraintStackTpl<Scalar> &constraints) {
  std::vector<xyz::polymorphic<ConstraintSetTpl<Scalar>>> components;
  for (size_t i = 0; i < constraints.size(); i++) {
    components.push_back(constraints.sets[i]);
  }
  return ConstraintSetProductTpl<Scalar>{components, constraints.dims()};
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
  using KnotType = gar::LqrKnotTpl<Scalar>;
  using ConstraintSetProduct = ConstraintSetProductTpl<Scalar>;
  using BlkJacobianType = BlkMatrix<MatrixXs, -1, 2>; // jacobians
  using LqrProblemType = gar::LqrProblemTpl<Scalar>;

  using Base::dyn_slacks;
  using Base::nsteps;
  using Base::problem_data;

  LqrProblemType lqr_problem; //< Linear-quadratic subproblem

  /// @name Lagrangian Gradients
  /// @{
  std::vector<VectorXs> Lxs; //< State gradients
  std::vector<VectorXs> Lus; //< Control gradients
  std::vector<VectorXs> Lvs; //< Path multiplier gradients
  /// @}

  /// @name Trial primal-dual step
  /// @{
  using Base::trial_us;
  using Base::trial_xs;
  std::vector<VectorXs> trial_vs;
  std::vector<VectorXs> trial_lams;
  /// @}

  /// @name Path constraint AL multipliers.
  /// @{
  // used for linesearch
  std::vector<VectorXs> lams_plus;
  std::vector<VectorXs> vs_plus;
  std::vector<VectorXs> vs_pdal;
  /// @}

  /// Shifted constraints the projection operators should be applied to.
  std::vector<VectorXs> shifted_constraints;
  std::vector<VectorXs> cstr_lx_corr;
  std::vector<VectorXs> cstr_lu_corr;
  /// Projected path constraint Jacobians (used to symmetrize the LQ subproblem)
  std::vector<BlkJacobianType> cstr_proj_jacs;
  /// Masks for active constraint sets
  std::vector<VecBool> active_constraints;
  /// Cartesian products of the constraint sets of each stage.
  std::vector<ConstraintSetProduct> cstr_product_sets;

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

  explicit WorkspaceTpl()
      : Base() {}
  explicit WorkspaceTpl(const TrajOptProblemTpl<Scalar> &problem);

  WorkspaceTpl(const WorkspaceTpl &) = delete;
  WorkspaceTpl &operator=(const WorkspaceTpl &) = delete;

  WorkspaceTpl(WorkspaceTpl &&) = default;
  WorkspaceTpl &operator=(WorkspaceTpl &&) = default;

  void cycleAppend(const TrajOptProblemTpl<Scalar> &problem,
                   shared_ptr<StageDataTpl<Scalar>> data);

  template <typename T>
  friend std::ostream &operator<<(std::ostream &oss,
                                  const WorkspaceTpl<T> &self);
};

template <typename Scalar>
std::ostream &operator<<(std::ostream &oss, const WorkspaceTpl<Scalar> &self) {
  return oss << fmt::format("{}", self);
}

} // namespace aligator

template <typename Scalar>
struct fmt::formatter<aligator::WorkspaceTpl<Scalar>> {
  constexpr auto parse(format_parse_context &ctx) const
      -> decltype(ctx.begin()) {
    return ctx.end();
  }

  auto format(const aligator::WorkspaceTpl<Scalar> &ws,
              format_context &ctx) const -> decltype(ctx.out()) {
    return fmt::format_to(ctx.out(),
                          "Workspace {{"
                          "\n  nsteps:       \t{:d}"
                          "\n  n_multipliers:\t{:d}"
                          "\n}}",
                          ws.nsteps, ws.dlams.size());
  }
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./workspace.txx"
#endif
