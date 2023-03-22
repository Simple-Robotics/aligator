/// @file solver-proxddp.hpp
/// @brief  Definitions for the proximal trajectory optimization algorithm.
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/merit-function.hpp"
#include "proxddp/core/proximal-penalty.hpp"
#include "proxddp/core/linesearch.hpp"
#include "proxddp/core/helpers-base.hpp"
#include "proxddp/core/enums.hpp"
#include "proxddp/utils/exceptions.hpp"
#include "proxddp/utils/logger.hpp"
#include "proxddp/utils/forward-dyn.hpp"

#include <proxnlp/constraint-base.hpp>
#include <proxnlp/bcl-params.hpp>

namespace proxddp {

using proxnlp::BCLParams;

/// @brief A proximal, augmented Lagrangian-type solver for trajectory
/// optimization.
template <typename _Scalar> struct SolverProxDDP {
public:
  // typedefs

  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using BlockXs = typename MatrixXs::BlockXpr;
  using Problem = TrajOptProblemTpl<Scalar>;
  using Workspace = WorkspaceTpl<Scalar>;
  using Results = ResultsTpl<Scalar>;
  using FunctionData = FunctionDataTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;
  using StageModel = StageModelTpl<Scalar>;
  using ConstraintType = StageConstraintTpl<Scalar>;
  using StageData = StageDataTpl<Scalar>;
  using VParams = value_function<Scalar>;
  using QParams = q_function<Scalar>;
  using ProxPenaltyType = ProximalPenaltyTpl<Scalar>;
  using ProxData = typename ProxPenaltyType::Data;
  using CallbackPtr = shared_ptr<helpers::base_callback<Scalar>>;
  using ConstraintStack = ConstraintStackTpl<Scalar>;
  using CstrSet = ConstraintSetBase<Scalar>;
  using TrajOptData = TrajOptDataTpl<Scalar>;
  using LinesearchOptions = typename Linesearch<Scalar>::Options;
  using CstrALWeightStrat = ConstraintALWeightStrategy<Scalar>;
  using LinesearchType = proxnlp::ArmijoLinesearch<Scalar>;

  std::vector<ProxPenaltyType> prox_penalties_;
  /// Subproblem tolerance
  Scalar inner_tol_;
  /// Desired primal feasibility
  Scalar prim_tol_;
  /// Solver tolerance \f$\epsilon > 0\f$.
  Scalar target_tol_ = 1e-6;

  Scalar mu_init = 0.01;
  Scalar rho_init = 0.;

  //// Inertia-correcting heuristic

  Scalar reg_min = 1e-10;
  Scalar reg_max = 1e9;
  Scalar reg_init = 1e-9;
  Scalar reg_init_nonzero = 1e-4;
  Scalar reg_inc_k = 8.;
  Scalar reg_inc_critical = 100.;
  Scalar reg_dec_k = 1. / 3.;

  Scalar xreg_ = reg_init;
  Scalar ureg_ = xreg_;

  //// Initial BCL tolerances

  Scalar inner_tol0 = 1.;
  Scalar prim_tol0 = 1.;

  /// Logger
  BaseLogger logger{};
#ifndef NDEBUG
  bool dump_linesearch_plot = false;
#endif

  /// Solver verbosity level.
  VerboseLevel verbose_;
  /// Type of Hessian approximation. Default is Gauss-Newton.
  HessianApprox hess_approx_ = HessianApprox::GAUSS_NEWTON;
  /// Linesearch options, as in proxnlp.
  LinesearchOptions ls_params;
  /// Type of linesearch strategy. Default is Armijo.
  LinesearchStrategy ls_strat = LinesearchStrategy::ARMIJO;
  /// Type of Lagrange multiplier update.
  MultiplierUpdateMode multiplier_update_mode = MultiplierUpdateMode::NEWTON;
  /// Linesearch mode.
  LinesearchMode ls_mode = LinesearchMode::PRIMAL;
  /// Weight of the dual variables in the primal-dual linesearch.
  Scalar dual_weight = 1.0;
  /// Type of rollout for the forward pass.
  RolloutType rollout_type_;
  /// Parameters for the BCL outer loop of the augmented Lagrangian algorithm.
  BCLParams<Scalar> bcl_params;

  /// Force the initial state @f$ x_0 @f$ to be fixed to the problem initial
  /// condition.
  bool is_x0_fixed_;

  /// @name Linear algebra options
  /// \{
  /// Maximum number of linear system refinement iterations
  std::size_t max_refinement_steps_ = 0;
  /// Target tolerance for solving the KKT system.
  Scalar refinement_threshold_ = 1e-13;
  /// Choice of factorization routine.
  LDLTChoice ldlt_algo_choice_;
  /// \}

  /// Maximum number \f$N_{\mathrm{max}}\f$ of Newton iterations.
  std::size_t max_iters;
  /// Maximum number of ALM iterations.
  std::size_t max_al_iters = 100;

  /// Minimum possible penalty parameter.
  Scalar MU_MIN = 1e-8;

  /// Callbacks
  std::vector<CallbackPtr> callbacks_;

  Workspace workspace_;
  Results results_;

  SolverProxDDP(const Scalar tol = 1e-6, const Scalar mu_init = 0.01,
                const Scalar rho_init = 0., const std::size_t max_iters = 1000,
                VerboseLevel verbose = VerboseLevel::QUIET,
                HessianApprox hess_approx = HessianApprox::GAUSS_NEWTON);

  PROXDDP_DEPRECATED const Results &getResults() { return results_; }
  PROXDDP_DEPRECATED const Workspace &getWorkspace() { return workspace_; }

  /// @brief Compute the linear search direction, i.e. the (regularized) SQP
  /// step.
  ///
  /// @pre This function assumes \f$\delta x_0\f$ has already been computed!
  /// @returns This computes the primal-dual step \f$(\delta \bfx,\delta
  /// \bfu,\delta\bmlam)\f$
  void linearRollout(const Problem &problem, Workspace &workspace,
                     const Results &results) const;

  /// @brief    Try a step of size \f$\alpha\f$.
  /// @returns  A primal-dual trial point
  ///           \f$(\bfx \oplus\alpha\delta\bfx, \bfu+\alpha\delta\bfu,
  ///           \bmlam+\alpha\delta\bmlam)\f$
  /// @returns  The trajectory cost.
  Scalar forward_linear_impl(const Problem &problem, Workspace &workspace,
                             const Results &results, const Scalar alpha) const;

  /// @brief    Policy rollout using the full nonlinear dynamics. The feedback
  /// gains need to be computed first. This will evaluate all the terms in the
  /// problem into the problem data, similar to TrajOptProblemTpl::evaluate().
  /// @returns  The trajectory cost.
  Scalar nonlinear_rollout_impl(const Problem &problem, Workspace &workspace,
                                const Results &results,
                                const Scalar alpha) const;

  Scalar forwardPass(const Problem &problem, Workspace &workspace,
                     const Results &results, const Scalar alpha);

  void compute_dir_x0(const Problem &problem, Workspace &workspace,
                      const Results &results) const;

  /// @brief    Terminal node.
  void computeTerminalValue(const Problem &problem, Workspace &workspace,
                            Results &results) const;

  /// @brief    Perform the Riccati backward pass.
  ///
  /// @pre  Compute the derivatives first!
  bool backwardPass(const Problem &problem, Workspace &workspace,
                    Results &results) const;

  /// @brief Allocate new workspace and results instances according to the
  /// specifications of @p problem.
  /// @param problem  The problem instance with respect to which memory will be
  /// allocated.
  void setup(const Problem &problem);

  void computeProxTerms(const std::vector<VectorXs> &xs,
                        const std::vector<VectorXs> &us,
                        Workspace &workspace) const;

  void computeProxDerivatives(const std::vector<VectorXs> &xs,
                              const std::vector<VectorXs> &us,
                              Workspace &workspace) const;

  /// Compute the Hamiltonian parameters at time @param t.
  void updateHamiltonian(const Problem &problem, const Results &results,
                         Workspace &workspace, const std::size_t) const;

  /// @brief Run the numerical solver.
  /// @param problem  The trajectory optimization problem to solve.
  /// @param xs_init  Initial trajectory guess.
  /// @param us_init  Initial control sequence guess.
  /// @param lams_init  Initial multiplier guess.
  /// @pre  You must call SolverProxDDP::setup beforehand to allocate a
  /// workspace and results.
  bool run(const Problem &problem, const std::vector<VectorXs> &xs_init = {},
           const std::vector<VectorXs> &us_init = {},
           const std::vector<VectorXs> &lams_init = {});

  /// @brief    Perform the inner loop of the algorithm (augmented Lagrangian
  /// minimization).
  bool innerLoop(const Problem &problem, Workspace &workspace,
                 Results &results);

  /// @brief    Compute the primal infeasibility measures.
  /// @warning  This will alter the constraint values (by projecting on the
  /// normal cone in-place).
  ///           Compute anything which accesses these before!
  void computeInfeasibilities(const Problem &problem, Workspace &workspace,
                              Results &results) const;

  void computeCriterion(const Problem &problem, Workspace &workspace,
                        Results &results) const;

  /// @name callbacks
  /// \{

  /// @brief    Add a callback to the solver instance.
  void registerCallback(const CallbackPtr &cb) { callbacks_.push_back(cb); }

  /// @brief    Remove all callbacks from the instance.
  void clearCallbacks() noexcept { callbacks_.clear(); }

  /// @brief    Invoke callbacks.
  void invokeCallbacks(Workspace &workspace, Results &results) {
    for (auto &cb : callbacks_) {
      cb->call(workspace, results);
    }
  }
  /// \}

  /// Evaluate the ALM/pdALM multiplier estimates.
  void computeMultipliers(const Problem &problem, Workspace &workspace,
                          const std::vector<VectorXs> &lams) const;

  void projectJacobians(const Problem &problem, Workspace &workspace) const;

  /// @copydoc mu_penal_
  PROXDDP_INLINE Scalar mu() const { return mu_penal_; }

  /// @copydoc mu_inverse_
  PROXDDP_INLINE Scalar mu_inv() const { return mu_inverse_; }

  /// @copydoc rho_penal_
  PROXDDP_INLINE Scalar rho() const { return rho_penal_; }

  //// Scaled variants

  /// @brief  Put together the Q-function parameters and compute the Riccati
  /// gains.
  PROXDDP_INLINE bool computeGains(const Problem &problem, Workspace &workspace,
                                   Results &results,
                                   const std::size_t step) const;

  auto getLinesearchMuLowerBound() const { return min_mu_linesearch_; }
  void setLinesearchMuLowerBound(Scalar mu) { min_mu_linesearch_ = mu; }
  /// @brief  Get the penalty parameter for linesearch.
  auto getLinesearchMu() const { return std::max(mu(), min_mu_linesearch_); }

protected:
  void update_tols_on_failure();
  void update_tols_on_success();

  /// Set dual proximal/ALM penalty parameter.
  PROXDDP_INLINE void set_penalty_mu(Scalar new_mu) noexcept {
    mu_penal_ = std::max(new_mu, MU_MIN);
    mu_inverse_ = 1. / new_mu;
  }

  PROXDDP_INLINE void set_rho(Scalar new_rho) noexcept { rho_penal_ = new_rho; }

  /// Update the dual proximal penalty according to BCL.
  PROXDDP_INLINE void bcl_update_alm_penalty() noexcept {
    set_penalty_mu(mu_penal_ * bcl_params.mu_update_factor);
  }

  /// Increase Tikhonov regularization.
  inline void increase_reg() {
    if (xreg_ == 0.) {
      xreg_ = reg_min;
    } else {
      xreg_ *= 10.;
      xreg_ = std::min(xreg_, reg_max);
    }
    ureg_ = xreg_;
  }

  /// Decrease Tikhonov regularization.
  inline void decrease_reg() {
    xreg_ *= 0.1;
    if (xreg_ < reg_min) {
      xreg_ = 0.;
    }
    ureg_ = xreg_;
  }

private:
  /// Dual proximal/ALM penalty parameter \f$\mu\f$
  /// This is the global parameter: scales may be applied for stagewise
  /// constraints, dynamicals...
  Scalar mu_penal_ = mu_init;
  Scalar min_mu_linesearch_ = 1e-8;
  /// Inverse ALM penalty parameter.
  Scalar mu_inverse_ = 1. / mu_penal_;
  /// Primal proximal parameter \f$\rho > 0\f$
  Scalar rho_penal_ = rho_init;
  /// Linesearch function
  LinesearchType linesearch_;
};

} // namespace proxddp

#include "proxddp/core/solver-proxddp.hxx"

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxddp/core/solver-proxddp.txx"
#endif
