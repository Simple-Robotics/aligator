/// @file solver-proxddp.hpp
/// @brief  Definitions for the proximal trajectory optimization algorithm.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/solver-util.hpp"
#include "proxddp/core/merit-function.hpp"
#include "proxddp/core/proximal-penalty.hpp"
#include "proxddp/core/linesearch.hpp"
#include "proxddp/core/helpers-base.hpp"
#include "proxddp/utils/exceptions.hpp"
#include "proxddp/utils/logger.hpp"
#include "proxddp/utils/rollout.hpp"

#include <proxnlp/constraint-base.hpp>

namespace proxddp {

enum class MultiplierUpdateMode { NEWTON, PRIMAL, PRIMAL_DUAL };

enum class RolloutType { LINEAR, NONLINEAR };

template <typename Scalar> struct BCLParams {

  /// Log-factor \f$\alpha_\eta\f$ for primal tolerance (failure)
  Scalar prim_alpha = 0.1;
  /// Log-factor \f$\beta_\eta\f$ for primal tolerance (success)
  Scalar prim_beta = 0.9;
  /// Log-factor \f$\alpha_\eta\f$ for dual tolerance (failure)
  Scalar dual_alpha = 1.;
  /// Log-factor \f$\beta_\eta\f$ for dual tolerance (success)
  Scalar dual_beta = 1.;
  /// Scale factor for the dual proximal penalty.
  Scalar mu_update_factor = 0.01;
  /// Scale factor for the primal proximal penalty.
  Scalar rho_update_factor = 0.1;
};

/// @brief Solver.
template <typename _Scalar> struct SolverProxDDP {
public:
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using BlockXs = typename MatrixXs::BlockXpr;
  using Problem = TrajOptProblemTpl<Scalar>;
  using Workspace = WorkspaceTpl<Scalar>;
  using Results = ResultsTpl<Scalar>;
  using FunctionData = FunctionDataTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;
  using StageModel = StageModelTpl<Scalar>;
  using Constraint = StageConstraintTpl<Scalar>;
  using StageData = StageDataTpl<Scalar>;
  using VParams = internal::value_storage<Scalar>;
  using QParams = internal::q_storage<Scalar>;
  using ProxPenaltyType = ProximalPenaltyTpl<Scalar>;
  using ProxData = typename ProxPenaltyType::Data;
  using CallbackPtr = shared_ptr<helpers::base_callback<Scalar>>;
  using ConstraintStack = ConstraintStackTpl<Scalar>;
  using LSOptions = typename proxnlp::Linesearch<Scalar>::Options;

  std::vector<ProxPenaltyType> prox_penalties_;
  /// Subproblem tolerance
  Scalar inner_tol_;
  /// Desired primal feasibility
  Scalar prim_tol_;

  /// Solver tolerance \f$\epsilon > 0\f$.
  Scalar target_tol_ = 1e-6;

  Scalar mu_init = 0.01;
  Scalar rho_init = 0.;

  Scalar reg_min = 1e-9;
  Scalar reg_max = 1e9;
  Scalar reg_init = 1e-9;
  Scalar xreg_ = reg_init;
  Scalar ureg_ = xreg_;

  Scalar inner_tol0 = 1.;
  Scalar prim_tol0 = 1.;

  ::proxddp::BaseLogger logger{};
#ifndef NDEBUG
  bool dump_linesearch_plot = false;
#endif

  VerboseLevel verbose_;
  /// Linesearch options, as in proxnlp.
  LSOptions ls_params;
  LinesearchStrategy ls_strat = LinesearchStrategy::ARMIJO;
  MultiplierUpdateMode multiplier_update_mode = MultiplierUpdateMode::NEWTON;
  LinesearchMode ls_mode = LinesearchMode::PRIMAL_DUAL;
  /// @brief Weight of the dual variables in the primal-dual linesearch.
  Scalar dual_weight = 1.0;
  RolloutType rollout_type = RolloutType::LINEAR;
  BCLParams<Scalar> bcl_params;

  /// Maximum number \f$N_{\mathrm{max}}\f$ of Newton iterations.
  std::size_t MAX_ITERS;
  /// Maximum number of ALM iterations.
  std::size_t MAX_AL_ITERS = MAX_ITERS;

  /// Minimum possible tolerance asked from the solver.
  Scalar TOL_MIN = 1e-8;
  /// Minimum possible penalty parameter.
  Scalar MU_MIN = 1e-8;

  /// Callbacks
  std::vector<CallbackPtr> callbacks_;

  std::unique_ptr<Workspace> workspace_;
  std::unique_ptr<Results> results_;

  Results &getResults() { return *results_; }
  Workspace &getWorkspace() { return *workspace_; }

  SolverProxDDP(const Scalar tol = 1e-6, const Scalar mu_init = 0.01,
                const Scalar rho_init = 0., const std::size_t max_iters = 1000,
                const VerboseLevel verbose = VerboseLevel::QUIET);

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
  void tryStep(const Problem &problem, Workspace &workspace,
               const Results &results, const Scalar alpha) const;

  /// @brief    Policy rollout using the full nonlinear dynamics. The feedback
  /// gains need to be computed first.
  void nonlinearRollout(const Problem &problem, Workspace &workspace,
                        const Results &results, const Scalar alpha) const {
    const std::size_t nsteps = workspace.nsteps;
    std::vector<VectorXs> &xs = workspace.trial_xs;
    std::vector<VectorXs> &us = workspace.trial_us;
    std::vector<VectorXs> &lams = workspace.trial_lams;
    TrajOptDataTpl<Scalar> &pd = workspace.trial_prob_data;

    computeDirX0(problem, workspace, results);

    {
      const StageModel &stage = *problem.stages_[0];
      stage.xspace().integrate(results.xs[0], alpha * workspace.dxs[0], xs[0]);
      lams[0] = results.lams[0] + alpha * workspace.dlams[0];
    }
#ifndef NDEBUG
    std::FILE *fi = std::fopen("pddp.log", "a");
#endif

    for (std::size_t i = 0; i < nsteps; i++) {
      const StageModel &stage = *problem.stages_[i];
      StageData &data = pd.getStageData(i);
      const int nu = stage.nu();
      const int ndual = stage.numDual();
      const int ndx2 = stage.ndx2();

      auto ff = results.getFeedforward(i);
      auto fb = results.getFeedback(i);
      auto ff_u = ff.head(nu);
      auto fb_u = fb.topRows(nu);
      auto ff_lm = ff.tail(ndual);
      auto fb_lm = fb.bottomRows(ndual);

      const VectorRef &dx = workspace.dxs[i];
      VectorRef &du = workspace.dus[i];
      du.head(nu) = alpha * ff_u + fb_u * dx;
      stage.uspace().integrate(results.us[i], du, us[i]);

      VectorRef &dlam = workspace.dlams[i + 1];
      dlam.head(ndual) = alpha * ff_lm + fb_lm * dx;
      lams[i + 1].head(ndual) = results.lams[i + 1] + dlam;

      const DynamicsModelTpl<Scalar> &dm = stage.dyn_model();
      DynamicsDataTpl<Scalar> &dd = data.dyn_data();
      const ConstraintStack &cstr_mgr = stage.constraints_;
      const ConstVectorRef dynlam =
          cstr_mgr.getConstSegmentByConstraint(lams[i + 1], 0);
      const ConstVectorRef dynprevlam =
          cstr_mgr.getConstSegmentByConstraint(workspace.prev_lams[i + 1], 0);
      VectorXs gap = this->mu_scaled() * (dynprevlam - dynlam);
      forwardDynamics(dm, xs[i], us[i], dd, xs[i + 1], 1, gap);

      VectorRef dx_next = workspace.dxs[i + 1].head(ndx2);
      stage.xspace_next().difference(results.xs[i + 1], xs[i + 1], dx_next);

      PROXDDP_RAISE_IF_NAN_NAME(xs[i + 1], fmt::format("xs[{:d}]", i + 1));
      PROXDDP_RAISE_IF_NAN_NAME(us[i], fmt::format("us[{:d}]", i));
      PROXDDP_RAISE_IF_NAN_NAME(lams[i + 1], fmt::format("lams[{:d}]", i + 1));
    }
    if (problem.term_constraint_) {
      VectorRef &dlam = workspace.dlams.back();
      const VectorRef &dx = workspace.dxs.back();
      auto ff = results.getFeedforward(nsteps);
      auto fb = results.getFeedback(nsteps);
      dlam = alpha * ff + fb * dx;
      lams.back() = results.lams.back() + dlam;
    }
#ifndef NDEBUG
    std::fclose(fi);
#endif
  }

  void computeDirX0(const Problem &problem, Workspace &workspace,
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
                        Workspace &workspace) const {
    const std::size_t nsteps = workspace.nsteps;
    for (std::size_t i = 0; i < nsteps; i++) {
      prox_penalties_[i].evaluate(xs[i], us[i], *workspace.prox_datas[i]);
    }
    prox_penalties_[nsteps].evaluate(xs[nsteps], us[nsteps - 1],
                                     *workspace.prox_datas[nsteps]);
  }

  void computeProxDerivatives(const std::vector<VectorXs> &xs,
                              const std::vector<VectorXs> &us,
                              Workspace &workspace) const {
    const std::size_t nsteps = workspace.nsteps;
    for (std::size_t i = 0; i < nsteps; i++) {
      prox_penalties_[i].computeGradients(xs[i], us[i],
                                          *workspace.prox_datas[i]);
      prox_penalties_[i].computeHessians(xs[i], us[i],
                                         *workspace.prox_datas[i]);
    }
    prox_penalties_[nsteps].computeGradients(xs[nsteps], us[nsteps - 1],
                                             *workspace.prox_datas[nsteps]);
    prox_penalties_[nsteps].computeHessians(xs[nsteps], us[nsteps - 1],
                                            *workspace.prox_datas[nsteps]);
  }

  /// Compute the Hamiltonian parameters at time @param t.
  void updateHamiltonian(const Problem &problem, const std::size_t t,
                         const Results &results, Workspace &workspace) const;

  /// @brief Run the numerical solver.
  /// @param problem  The trajectory optimization problem to solve.
  /// @param xs_init  Initial trajectory guess.
  /// @param us_init  Initial control sequence guess.
  /// @param lams_init  Initial multiplier guess.
  /// @pre  You must call SolverProxDDP::setup beforehand to allocate a
  /// workspace and results.
  bool run(const Problem &problem,
           const std::vector<VectorXs> &xs_init = DEFAULT_VECTOR<Scalar>,
           const std::vector<VectorXs> &us_init = DEFAULT_VECTOR<Scalar>,
           const std::vector<VectorXs> &lams_init = DEFAULT_VECTOR<Scalar>);

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

  /// @name callbacks
  /// \{

  /// @brief    Add a callback to the solver instance.
  void registerCallback(const CallbackPtr &cb) { callbacks_.push_back(cb); }

  /// @brief    Remove all callbacks from the instance.
  void clearCallbacks() { callbacks_.clear(); }

  /// @brief    Invoke callbacks.
  void invokeCallbacks(Workspace &workspace, Results &results) {
    for (auto &cb : callbacks_) {
      cb->call(workspace, results);
    }
  }
  /// \}

  /// Evaluate the ALM/pdALM multiplier estimates.
  void computeMultipliers(const Problem &problem, Workspace &workspace,
                          const std::vector<VectorXs> &lams,
                          TrajOptDataTpl<Scalar> &pd,
                          bool update_jacobians = false) const {
    ;
    using CstrSet = ConstraintSetBase<Scalar>;
    const std::size_t nsteps = workspace.nsteps;

    std::vector<VectorXs> &lams_plus = workspace.lams_plus;
    std::vector<VectorXs> &lams_pdal = workspace.lams_pdal;

    {
      const VectorXs &lam0 = lams[0];
      const VectorXs &plam0 = workspace.prev_lams[0];
      shared_ptr<CstrSet> cstr =
          std::make_shared<proxnlp::EqualityConstraint<Scalar>>();
      FunctionData &data = pd.getInitData();
      auto expr = plam0 + mu_inv() * data.value_;
      cstr->normalConeProjection(expr, lams_plus[0]);
      lams_pdal[0] = (1 + dual_weight) * lams_plus[0] - dual_weight * lam0;
      if (update_jacobians)
        cstr->applyNormalConeProjectionJacobian(expr, data.jac_buffer_);
    }

    if (problem.term_constraint_) {
      const VectorXs &lamN = lams.back();
      const VectorXs &plamN = workspace.prev_lams.back();
      const Constraint &termcstr = problem.term_constraint_.get();
      const CstrSet &cstr = *termcstr.set;
      FunctionData &data = *pd.term_cstr_data;
      auto expr = plamN + mu_inv() * data.value_;
      cstr.normalConeProjection(expr, lams_plus.back());
      lams_pdal.back() =
          (1 + dual_weight) * lams_plus.back() - dual_weight * lamN;
      if (update_jacobians)
        cstr.applyNormalConeProjectionJacobian(expr, data.jac_buffer_);
    }

    // loop over the stages
    for (std::size_t i = 0; i < nsteps; i++) {
      const StageModel &stage = *problem.stages_[i];
      const StageData &sdata = pd.getStageData(i);
      const ConstraintStack &mgr = stage.constraints_;

      // enumerate the constraints and perform projection
      auto cstr_callback = [&](auto mgr, std::size_t k, const VectorXs &lami,
                               const VectorXs &plami, VectorXs &lamplusi,
                               VectorXs &lampdali) {
        const auto lami_k = mgr.getConstSegmentByConstraint(lami, k);
        const auto plami_k = mgr.getConstSegmentByConstraint(plami, k);
        auto lamplus_k = mgr.getSegmentByConstraint(lamplusi, k);
        auto lampdal_k = mgr.getSegmentByConstraint(lampdali, k);

        const CstrSet &set = mgr.getConstraintSet(k);
        FunctionData &data = *sdata.constraint_data[k];
        auto expr = plami_k + mu_inv_scaled() * data.value_;
        set.normalConeProjection(expr, lamplus_k);
        lampdal_k = (1 + dual_weight) * lamplus_k - dual_weight * lami_k;
        if (update_jacobians)
          set.applyNormalConeProjectionJacobian(expr, data.jac_buffer_);
      };

      for (std::size_t k = 0; k < mgr.numConstraints(); k++) {
        cstr_callback(mgr, k, lams[i + 1], workspace.prev_lams[i + 1],
                      lams_plus[i + 1], lams_pdal[i + 1]);
      }
    }
  }

  /// @copydoc mu_penal_
  inline Scalar mu() const { return mu_penal_; }

  /// @copydoc mu_inverse_
  inline Scalar mu_inv() const { return mu_inverse_; }

  /// Scaled penalty parameter.
  Scalar mu_scaled() const { return mu(); }

  /// Scaled inverse penalty parameter.
  Scalar mu_inv_scaled() const { return 1. / mu_scaled(); }

  /// Proximal parameter.
  Scalar rho() const { return rho_penal_; }

protected:
  /// @brief  Put together the Q-function parameters and compute the Riccati
  /// gains.
  inline bool computeGains(const Problem &problem, Workspace &workspace,
                           Results &results, const std::size_t step) const;

  void updateTolerancesOnFailure() {
    prim_tol_ = prim_tol0 * std::pow(mu_penal_, bcl_params.prim_alpha);
    inner_tol_ = inner_tol0 * std::pow(mu_penal_, bcl_params.dual_alpha);
  }

  void updateTolerancesOnSuccess() {
    prim_tol_ = prim_tol_ * std::pow(mu_penal_, bcl_params.prim_beta);
    inner_tol_ = inner_tol_ * std::pow(mu_penal_, bcl_params.dual_beta);
  }

  /// Set dual proximal/ALM penalty parameter.
  void setPenalty(Scalar new_mu) {
    mu_penal_ = std::max(new_mu, MU_MIN);
    mu_inverse_ = 1. / new_mu;
  }

  void setRho(Scalar new_rho) { rho_penal_ = new_rho; }

  /// Update the dual proximal penalty according to BCL.
  void bclUpdateALPenalty() {
    setPenalty(mu_penal_ * bcl_params.mu_update_factor);
  }

  /// Increase Tikhonov regularization.
  void increase_reg() {
    if (xreg_ == 0.) {
      xreg_ = reg_min;
    } else {
      xreg_ *= 10.;
      xreg_ = std::min(xreg_, reg_max);
    }
    ureg_ = xreg_;
  }

  /// Decrease Tikhonov regularization.
  void decrease_reg() {
    xreg_ *= 0.1;
    if (xreg_ < reg_min) {
      xreg_ = 0.;
    }
    ureg_ = xreg_;
  }

private:
  /// Dual proximal/ALM penalty parameter \f$\mu\f$
  Scalar mu_penal_ = mu_init;
  /// Inverse ALM penalty parameter.
  Scalar mu_inverse_ = 1. / mu_penal_;
  /// Primal proximal parameter \f$\rho > 0\f$
  Scalar rho_penal_ = rho_init;
  /// Scale for the dynamical constraint
  Scalar mu_scale_0 = 0.01;
  PDALFunction<Scalar> merit_fun;
};

} // namespace proxddp

#include "proxddp/core/solver-proxddp.hxx"
