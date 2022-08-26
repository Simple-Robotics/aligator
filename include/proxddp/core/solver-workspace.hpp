/**
 * @file    solver-workspace.hpp
 * @brief   Define workspace for the ProxDDP solver.
 * @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
 */
#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/value-storage.hpp"
#include "proxddp/core/proximal-penalty.hpp"

namespace proxddp {

template <typename Scalar> struct WorkspaceBaseTpl {
  using value_storage_t = internal::value_storage<Scalar>;
  using q_storage_t = internal::q_storage<Scalar>;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  /// Number of steps in the problem.
  const std::size_t nsteps;
  /// Problem data.
  TrajOptDataTpl<Scalar> problem_data;
  /// @name Linesearch data
  /// Problem data instance for linesearch.
  TrajOptDataTpl<Scalar> trial_prob_data;
  std::vector<VectorXs> trial_xs_;
  std::vector<VectorXs> trial_us_;

  /// @name Value function and Hamiltonian.
  /// Value function parameter storage
  std::vector<value_storage_t> value_params;
  /// Q-function storage
  std::vector<q_storage_t> q_params;

  explicit WorkspaceBaseTpl(const TrajOptProblemTpl<Scalar> &problem)
      : nsteps(problem.numSteps()), problem_data(problem),
        trial_prob_data(problem) {
    trial_xs_.resize(nsteps + 1);
    trial_us_.resize(nsteps);
    xs_default_init(problem, trial_xs_);
    us_default_init(problem, trial_us_);
  }

  // virtual ~WorkspaceBaseTpl() = default;
};

/** @brief Workspace for the solver.
 *
 * @details This struct holds data for the Riccati forward and backward passes,
 *          the primal-dual steps, problem data...
 */
template <typename _Scalar> struct WorkspaceTpl : WorkspaceBaseTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using ProxPenalty = ProximalPenaltyTpl<Scalar>;
  using ProxData = typename ProxPenalty::Data;
  using StageModel = StageModelTpl<Scalar>;
  using Base = WorkspaceBaseTpl<Scalar>;

  using Base::nsteps;
  using Base::problem_data;
  using Base::q_params;
  using Base::trial_prob_data;
  using Base::trial_us_;
  using Base::trial_xs_;
  using Base::value_params;

  /// Proximal penalty data.
  std::vector<shared_ptr<ProxData>> prox_datas;

  /// Lagrange multipliers for ALM & linesearch.
  std::vector<VectorXs> trial_lams_;
  std::vector<VectorXs> lams_plus;
  std::vector<VectorXs> lams_pdal;

  /// @name Riccati gains and buffers for primal-dual steps

  std::vector<VectorXs> pd_step_;
  std::vector<VectorRef> dxs_;
  std::vector<VectorRef> dus_;
  std::vector<VectorRef> dlams_;

  /// Buffer for KKT matrix
  MatrixXs kkt_matrix_buf_;
  /// Buffer for KKT right hand side
  MatrixXs kkt_rhs_buf_;

  /// @name Previous proximal iterates

  std::vector<VectorXs> prev_xs;
  std::vector<VectorXs> prev_us;
  std::vector<VectorXs> prev_lams;

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
    return kkt_matrix_buf_.topLeftCorner(nprim + ndual, nprim + ndual);
  }

  Eigen::Block<MatrixXs, -1, -1> getKktRhs(const int nprim, const int ndual,
                                           const int ndx1) {
    return kkt_rhs_buf_.topLeftCorner(nprim + ndual, ndx1 + 1);
  }

  template <typename T>
  friend std::ostream &operator<<(std::ostream &oss,
                                  const WorkspaceTpl<T> &self);
};

} // namespace proxddp

#include "proxddp/core/solver-workspace.hxx"
