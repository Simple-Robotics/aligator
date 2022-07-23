/**
 * @file    solver-workspace.hpp
 * @brief   Define workspace for the ProxDDP solver.
 * @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
 */
#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/value_storage.hpp"
#include "proxddp/core/proximal-penalty.hpp"

namespace proxddp {

/** @brief Workspace for the solver.
 *
 * @details This struct holds data for the Riccati forward and backward passes,
 *          the primal-dual steps, problem data...
 */
template <typename _Scalar> struct WorkspaceTpl {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using value_storage_t = internal::value_storage<Scalar>;
  using q_storage_t = internal::q_function_storage<Scalar>;
  using ProxPenalty = ProximalPenaltyTpl<Scalar>;
  using ProxData = typename ProxPenalty::Data;
  using StageModel = StageModelTpl<Scalar>;

  const std::size_t nsteps;

  TrajOptDataTpl<Scalar> problem_data;
  TrajOptDataTpl<Scalar> trial_prob_data;

  /// Value function parameter storage
  std::vector<value_storage_t> value_params;

  /// Q-function storage
  std::vector<q_storage_t> q_params;

  std::vector<shared_ptr<ProxData>> prox_datas;

  std::vector<VectorXs> lams_plus_;
  std::vector<VectorXs> lams_pdal_;

  /// @name Riccati gains and buffers for primal-dual steps

  std::vector<VectorXs> pd_step_;
  std::vector<VectorRef> dxs_;
  std::vector<VectorRef> dus_;
  std::vector<VectorRef> dlams_;

  /// Buffer for KKT matrix
  MatrixXs kktMatrixFull_;
  /// Buffer for KKT right hand side
  MatrixXs kktRhsFull_;

  std::vector<VectorXs> trial_xs_;
  std::vector<VectorXs> trial_us_;
  std::vector<VectorXs> trial_lams_;

  /// @name Previous proximal iterates

  std::vector<VectorXs> prev_xs_;
  std::vector<VectorXs> prev_us_;
  std::vector<VectorXs> prev_lams_;

  /// Subproblem termination criterion for each stage.
  VectorXs inner_criterion_by_stage;
  /// Constraint violation for each stage of the TrajOptProblemTpl.
  VectorXs primal_infeas_by_stage;
  /// Dual infeasibility for each stage of the TrajOptProblemTpl.
  VectorXs dual_infeas_by_stage;

  /// Overall subproblem termination criterion.
  Scalar inner_criterion = 0.;

  explicit WorkspaceTpl(const TrajOptProblemTpl<Scalar> &problem);

  Eigen::Block<MatrixXs, -1, -1> getKktView(const int nprim, const int ndual) {
    return kktMatrixFull_.topLeftCorner(nprim + ndual, nprim + ndual);
  }

  Eigen::Block<MatrixXs, -1, -1> getKktRhs(const int nprim, const int ndual,
                                           const int ndx1) {
    return kktRhsFull_.topLeftCorner(nprim + ndual, ndx1 + 1);
  }

  template <typename T>
  friend std::ostream &operator<<(std::ostream &oss,
                                  const WorkspaceTpl<T> &self);
};

} // namespace proxddp

#include "proxddp/core/solver-workspace.hxx"
