#pragma once

#include "proxddp/core/solver-util.hpp"
#include "proxddp/core/solver-results.hpp"
#include "proxddp/core/linesearch.hpp"
#include "proxddp/core/helpers-base.hpp"
#include "proxddp/core/explicit-dynamics.hpp"

#include "proxddp/fddp/results.hpp"
#include "proxddp/fddp/workspace.hpp"
#include "proxddp/fddp/linesearch.hpp"

#include "proxddp/utils/exceptions.hpp"
#include "proxddp/utils/logger.hpp"
#include "proxddp/utils/rollout.hpp"

#include <fmt/ostream.h>

#include <Eigen/Cholesky>

#define proxddp_fddp_warning(msg)                                              \
  fmt::print(fmt::fg(fmt::color::yellow), "[SolverFDDP] ({}) warning: {}",     \
             __FUNCTION__, msg)

namespace proxddp {

/**
 * @brief The feasible DDP (FDDP) algorithm
 *
 */
template <typename Scalar> struct SolverFDDP {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Problem = TrajOptProblemTpl<Scalar>;
  using StageModel = StageModelTpl<Scalar>;
  using StageData = StageDataTpl<Scalar>;
  using ProblemData = TrajOptDataTpl<Scalar>;
  using Results = ResultsFDDPTpl<Scalar>;
  using Workspace = WorkspaceFDDPTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using VParams = internal::value_storage<Scalar>;
  using QParams = internal::q_storage<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;
  using ExpModel = ExplicitDynamicsModelTpl<Scalar>;
  using ExpData = ExplicitDynamicsDataTpl<Scalar>;
  using CallbackPtr = shared_ptr<helpers::base_callback<Scalar>>;

  const Scalar tol_;

  /// @name Regularization parameters
  /// \{
  Scalar xreg_;
  Scalar ureg_ = xreg_;
  Scalar reg_min_ = 1e-9;
  Scalar reg_max_ = 1e9;
  /// Regularization decrease factor
  Scalar reg_dec_factor_ = 0.1;
  /// Regularization increase factor
  Scalar reg_inc_factor_ = 10.;
  /// \}

  Scalar th_grad_ = 1e-12;
  Scalar th_step_dec_ = 0.5;
  Scalar th_step_inc_ = 0.01;

  LinesearchParams<Scalar> ls_params;
  enum ls_types { GOLDSTEIN, ARMIJO };
  ls_types ls_type = GOLDSTEIN;

  VerboseLevel verbose_;
  /// Maximum number of iterations for the solver.
  std::size_t MAX_ITERS = 200;

  /// Callbacks
  std::vector<CallbackPtr> callbacks_;

  std::unique_ptr<Results> results_;
  std::unique_ptr<Workspace> workspace_;

  SolverFDDP(const Scalar tol = 1e-6,
             VerboseLevel verbose = VerboseLevel::QUIET,
             const Scalar reg_init = 1e-10);

  const Results &getResults() const { return *results_; }
  const Workspace &getWorkspace() const { return *workspace_; }

  /// @brief Allocate workspace and results structs.
  void setup(const Problem &problem);

  /**
   * @brief   Perform a nonlinear rollout, keeping an infeasibility gap.
   * @details Perform a nonlinear rollout using the computed sensitivity gains
   * from the backward pass, while keeping the dynamical feasibility gaps open
   * proportionally to the step-size @p alpha.
   * @param[in]   problem
   * @param[in]   results
   * @param[out]  workspace
   * @param[in]   alpha step-size.
   */
  static void forwardPass(const Problem &problem, const Results &results,
                          Workspace &workspace, const Scalar alpha);

  /// @brief Try a given step size, and store the resulting cost in
  /// `Results::traj_cost_`.
  static Scalar tryStep(const Problem &problem, const Results &results,
                        Workspace &workspace, const Scalar alpha);

  void computeDirectionalDerivatives(Workspace &workspace, Results &results,
                                     Scalar &d1, Scalar &d2) const;

  /// @brief  Correct the directional derivatives.
  static void directionalDerivativeCorrection(const Problem &problem,
                                              const Workspace &workspace,
                                              Results &results, Scalar &d1,
                                              Scalar &d2);

  /**
   * @brief   Computes dynamical feasibility gaps.
   * @details This computes the difference \f$x_{i+1} \ominus f(x_i, u_i)$, as
   * well as the residual of initial condition. This function will compute the
   * forward dynamics at every step to compute the forward map $f(x_i, u_i)$.
   */
  static Scalar computeInfeasibility(const Problem &problem,
                                     const std::vector<VectorXs> &xs,
                                     const std::vector<VectorXs> &us,
                                     Workspace &workspace);

  /// @brief   Perform the backward pass and compute Riccati gains.
  void backwardPass(const Problem &problem, Workspace &workspace,
                    Results &results) const;

  void increaseRegularization() {
    xreg_ *= reg_inc_factor_;
    xreg_ = std::min(xreg_, reg_max_);
    ureg_ = xreg_;
  }

  void decreaseRegularization() {
    xreg_ *= reg_dec_factor_;
    xreg_ = std::max(xreg_, reg_min_);
    ureg_ = xreg_;
  }

  /// @brief Compute the dual feasibility of the problem.
  void computeCriterion(Workspace &workspace, Results &results);

  /// @brief    Add a callback to the solver instance.
  void registerCallback(const CallbackPtr &cb) { callbacks_.push_back(cb); }

  /// @brief    Remove all callbacks from the instance.
  void clearCallbacks() { callbacks_.clear(); }

  void invokeCallbacks(Workspace &workspace, Results &results) {
    for (auto &cb : callbacks_) {
      cb->call(workspace, results);
    }
  }

  bool run(const Problem &problem,
           const std::vector<VectorXs> &xs_init = DEFAULT_VECTOR<Scalar>,
           const std::vector<VectorXs> &us_init = DEFAULT_VECTOR<Scalar>) {

    const Scalar fd_eps = 1e-9;

    if (results_ == 0 || workspace_ == 0) {
      proxddp_runtime_error(
          "Either results or workspace not allocated. Call setup() first!");
    }
    Results &results = *results_;
    Workspace &workspace = *workspace_;

    checkTrajectoryAndAssign(problem, xs_init, us_init, results.xs_,
                             results.us_);

    ::proxddp::BaseLogger logger{};
    if (verbose_ > 0)
      logger.start();

    // in Crocoddyl, linesearch xs is primed to use problem x0
    {
      const VectorXs &x0 = problem.getInitState();
      workspace.trial_xs_[0] = x0;
      workspace.xnexts_[0] = x0;
    }

    auto linesearch_fun = [&](const Scalar alpha) {
      return tryStep(problem, results, workspace, alpha);
    };

    LogRecord record;
    record.inner_crit = 0.;
    record.dual_err = 0.;
    record.dphi0 = 0.;
    std::size_t &iter = results.num_iters;
    for (iter = 0; iter < MAX_ITERS; ++iter) {

      record.iter = iter + 1;

      problem.evaluate(results.xs_, results.us_, workspace.problem_data);
      results.traj_cost_ =
          computeTrajectoryCost(problem, workspace.problem_data);
      // evaluate the forward rollout into workspace.xnexts_
      results.primal_infeasibility =
          computeInfeasibility(problem, results.xs_, results.us_, workspace);
      record.prim_err = results.primal_infeasibility;
      problem.computeDerivatives(results.xs_, results.us_,
                                 workspace.problem_data);

      backwardPass(problem, workspace, results);
      computeCriterion(workspace, results);

      PROXDDP_RAISE_IF_NAN(results.dual_infeasibility);
      record.dual_err = results.dual_infeasibility;
      record.merit = results.traj_cost_;
      record.inner_crit = 0.;
      record.xreg = xreg_;

      if (results.dual_infeasibility < tol_) {
        results.conv = true;
        break;
      }

      Scalar alpha_opt = 1;
      Scalar phi0 = results.traj_cost_;
      Scalar d1_phi, d2_phi;
      computeDirectionalDerivatives(workspace, results, d1_phi, d2_phi);
      PROXDDP_RAISE_IF_NAN(d1_phi);
      PROXDDP_RAISE_IF_NAN(d2_phi);
#ifndef NDEBUG
      {
        Scalar phi1 = linesearch_fun(fd_eps);
        assert(math::scalar_close(phi0, linesearch_fun(0.),
                                  std::numeric_limits<double>::epsilon()));
        Scalar finite_diff_d1 = (linesearch_fun(fd_eps) - phi0) / fd_eps;
        assert(
            math::scalar_close(finite_diff_d1, d1_phi, std::pow(fd_eps, 0.5)));
      }
#endif
      record.dphi0 = d1_phi;

      // quadratic model lambda; captures by copy
      auto ls_model = [=, &problem, &workspace, &results](const Scalar alpha) {
        Scalar d1 = d1_phi;
        Scalar d2 = d2_phi;
        directionalDerivativeCorrection(problem, workspace, results, d1, d2);
        return phi0 + alpha * (d1 + 0.5 * d2 * alpha);
      };

      bool d1_small = std::abs(d1_phi) < th_grad_;
      if (!d1_small) {
        switch (ls_type) {
        case ARMIJO:
          proxnlp::ArmijoLinesearch<Scalar>::run(
              linesearch_fun, phi0, d1_phi, verbose_, ls_params.ls_beta,
              ls_params.armijo_c1, ls_params.alpha_min, alpha_opt);
          break;
        case GOLDSTEIN:
          FDDPGoldsteinLinesearch<Scalar>::run(linesearch_fun, ls_model, phi0,
                                               verbose_, ls_params, th_grad_,
                                               alpha_opt);
          break;
        default:
          break;
        }
        record.step_size = alpha_opt;
      }
      // forwardPass(problem, results, workspace, alpha_opt);
      Scalar phi_new = tryStep(problem, results, workspace, alpha_opt);
      PROXDDP_RAISE_IF_NAN(phi_new);
      Scalar dphi = phi_new - phi0;
      record.dM = dphi;

      results.xs_ = workspace.trial_xs_;
      results.us_ = workspace.trial_us_;
      if (d1_small) {
        results.conv = true;
        if (verbose_ > 0)
          logger.log(record);
        break;
      }

      if (alpha_opt > th_step_dec_) {
        decreaseRegularization();
      }
      if (alpha_opt <= th_step_inc_) {
        increaseRegularization();
        if (xreg_ == reg_max_) {
          results.conv = false;
          break;
        }
      }

      invokeCallbacks(workspace, results);
      if (verbose_ > 0)
        logger.log(record);
    }

    logger.finish(results.conv);
    return results.conv;
  }

  static DynamicsDataTpl<Scalar> &
  stage_get_dynamics_data(StageDataTpl<Scalar> &sd) {
    return static_cast<DynamicsDataTpl<Scalar> &>(*sd.constraint_data[0]);
  }

  static const DynamicsDataTpl<Scalar> &
  stage_get_dynamics_data(const StageDataTpl<Scalar> &sd) {
    return static_cast<const DynamicsDataTpl<Scalar> &>(*sd.constraint_data[0]);
  }
};

} // namespace proxddp

#include "proxddp/fddp/solver-fddp.hxx"
