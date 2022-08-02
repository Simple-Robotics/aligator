#pragma once

#include "proxddp/core/solver-base.hpp"
#include "proxddp/core/solver-results.hpp"
#include "proxddp/core/linesearch.hpp"
#include "proxddp/core/helpers-base.hpp"
#include "proxddp/core/explicit-dynamics.hpp"

#include "proxddp/fddp/results.hpp"
#include "proxddp/fddp/workspace.hpp"
#include "proxddp/fddp/linsesearch.hpp"

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

  /// @brief    Evaluate the defects in the dynamics.
  /// @warning  We assume the dynamics were already computed.
  void evaluateGaps(const Problem &problem, const std::vector<VectorXs> &xs,
                    const std::vector<VectorXs> &us, const Workspace &workspace,
                    Results &results) const;

  /// @brief Try a given step size, and store the resulting cost in
  /// `Results::traj_cost_`.
  Scalar tryStep(const Problem &problem, const Results &results,
                 Workspace &workspace, const Scalar alpha) const;

  void computeDirectionalDerivatives(Workspace &workspace, Results &results,
                                     Scalar &d1, Scalar &d2) const {
    const std::size_t nsteps = workspace.nsteps;
    d1 = 0.; // cost directional derivative
    d2 = 0.;

    assert(workspace.q_params.size() == nsteps);
    assert(workspace.value_params.size() == (nsteps + 1));
    for (std::size_t i = 0; i < nsteps; i++) {
      const QParams &qpar = workspace.q_params[i];
      ConstVectorRef Qu = qpar.Qu_;
      ConstVectorRef ff = results.getFeedforward(i);
      d1 += Qu.dot(ff);
      d2 += ff.dot(workspace.Quuks_[i]);
    }
    for (std::size_t i = 0; i <= nsteps; i++) {
      // account for infeasibility
      const VParams &vpar = workspace.value_params[i];
      VectorXs &ftVxx = workspace.f_t_Vxx_[i];
      ftVxx = vpar.Vxx_ * workspace.feas_gaps_[i];
      d1 += vpar.Vx_.dot(workspace.feas_gaps_[i]);
      d2 = d2 - ftVxx.dot(workspace.feas_gaps_[i]);
    }
  }

  /// @brief  Correct the directional derivatives.
  static void directionalDerivativeCorrection(const Problem &problem,
                                              Workspace &workspace,
                                              Results &results, Scalar &d1,
                                              Scalar &d2) {
    const std::size_t nsteps = workspace.nsteps;
    const VectorOfVectors &xs = results.xs_;
    const VectorOfVectors &us = results.us_;

    Scalar dv = 0.;
    for (std::size_t i = 0; i < nsteps; i++) {
      const Manifold &space = problem.stages_[i]->xspace();
      space.difference(workspace.trial_xs_[i], xs[i], workspace.dxs_[i]);

      const VParams &vpar = workspace.value_params[i];
      const VectorXs &ftVxx = workspace.f_t_Vxx_[i];
      // ftVxx = vpar.Vxx_ * workspace.feas_gaps_[i]; // same as l.145
      dv += workspace.dxs_[i].dot(ftVxx);
    }

    d1 += dv;
    d2 += -2 * dv;
  }

  /**
   * @brief  Perform a nonlinear rollout, keeping an infeasibility gap.
   */
  static void forwardPass(const Problem &problem, const Results &results,
                          Workspace &workspace, const Scalar alpha);

  /**
   * @brief   Computes dynamical feasibility gaps.
   * @details This computes the difference \f$x_{i+1} \ominus \f$, as well as
   *          the residual of initial condition.
   *
   */
  static void computeInfeasibility(const Problem &problem, Results &results,
                                   Workspace &workspace);

  /**
   * @brief   Compute Riccati gains.
   */
  void backwardPass(const Problem &problem, Workspace &workspace,
                    Results &results) const {

    const std::size_t nsteps = workspace.nsteps;

    ProblemData &prob_data = workspace.problem_data;
    const CostData &term_cost_data = *prob_data.term_cost_data;
    VParams &term_value = workspace.value_params[nsteps];
    term_value.v_2() = 2 * term_cost_data.value_;
    term_value.Vx_ = term_cost_data.Lx_;
    term_value.Vxx_ = term_cost_data.Lxx_;
    term_value.Vxx_.diagonal().array() += xreg_;
    term_value.storage =
        term_value.storage.template selfadjointView<Eigen::Lower>();

    std::size_t i;
    for (std::size_t k = 0; k < nsteps; k++) {
      i = nsteps - k - 1;
      const VParams &vnext = workspace.value_params[i + 1];
      QParams &qparam = workspace.q_params[i];

      StageModel &sm = *problem.stages_[i];
      StageData &sd = prob_data.getData(i);

      const int nu = sm.nu();
      const int ndx1 = sm.ndx1();
      const int nt = ndx1 + nu;
      assert((qparam.storage.cols() == nt + 1) &&
             (qparam.storage.rows() == nt + 1));
      assert(qparam.grad_.size() == nt);

      const CostData &cd = *sd.cost_data;
      DynamicsDataTpl<Scalar> &dd = stage_get_dynamics_data(sd);

      /* Assemble Q-function */
      ConstMatrixRef J_x_u = dd.jac_buffer_.leftCols(ndx1 + nu);

      qparam.q_2() = 2 * cd.value_;
      qparam.grad_ = cd.grad_;
      qparam.hess_ = cd.hess_;

      // fmt::print("==== NODE t = {:d} ====\n", i);
      // fmt::print("vnext: {}\n", vnext);
      // fmt::print("vgrad: {}\n", vnext.Vx_.transpose());
      // TODO: implement second-order derivatives for the Q-function
      qparam.grad_ += J_x_u.transpose() * vnext.Vx_;
      qparam.hess_ += J_x_u.transpose() * vnext.Vxx_ * J_x_u;

      qparam.Quu_.diagonal().array() += ureg_;
      qparam.storage = qparam.storage.template selfadjointView<Eigen::Lower>();

      /* Compute gains */
      // MatrixXs &kkt_mat = workspace.kkt_matrix_bufs[i];
      MatrixXs &kkt_rhs = workspace.kkt_rhs_bufs[i];

      // kkt_mat = qparam.Quu_;
      VectorRef ffwd = results.getFeedforward(i);
      MatrixRef fback = results.getFeedback(i);
      ffwd = -qparam.Qu_;
      fback = -qparam.Qxu_.transpose();
      kkt_rhs = results.gains_[i];

      Eigen::LLT<MatrixXs> &llt = workspace.llts_[i];
      llt.compute(qparam.Quu_);
      llt.solveInPlace(results.gains_[i]);

      workspace.Quuks_[i] = qparam.Quu_ * ffwd;

      /* Compute value function */
      VParams &vcur = workspace.value_params[i];
      vcur.Vx_ = qparam.Qx_ + fback.transpose() * qparam.Qu_;
      vcur.Vxx_ = qparam.Qxx_ + qparam.Qxu_ * fback;
      vcur.Vx_.noalias() += vcur.Vxx_ * workspace.feas_gaps_[i + 1];
      vcur.Vxx_.diagonal().array() += xreg_;
      vcur.storage = vcur.storage.template selfadjointView<Eigen::Lower>();
    }
    assert(i == 0);
  }

  void increase_reg() {
    xreg_ *= reg_inc_factor_;
    xreg_ = std::min(xreg_, reg_max_);
    ureg_ = xreg_;
  }

  void decrease_reg() {
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
    for (auto cb : callbacks_) {
      cb->call(this, workspace, results);
    }
  }

  bool run(const Problem &problem,
           const std::vector<VectorXs> &xs_init = DEFAULT_VECTOR<Scalar>,
           const std::vector<VectorXs> &us_init = DEFAULT_VECTOR<Scalar>) {
    if (results_ == 0 || workspace_ == 0) {
      proxddp_runtime_error(
          "Either results or workspace not allocated. Call setup() first!");
    }
    Results &results = *results_;
    Workspace &workspace = *workspace_;

    checkTrajectoryAndAssign(problem, xs_init, us_init, results.xs_,
                             results.us_);

    ::proxddp::BaseLogger logger{};
    logger.start();

    // in Crocoddyl, linesearch xs is primed to use problem x0
    workspace.trial_xs_[0] = problem.getInitState();

    auto linesearch_fun = [&](const Scalar alpha) {
      return tryStep(problem, results, workspace, alpha);
    };

    forwardPass(problem, results, workspace, 1.);
    LogRecord record;
    record.inner_crit = 0.;
    record.dual_err = 0.;
    record.dphi0 = 0.;
    std::size_t &iter = results.num_iters;
    for (iter = 0; iter < MAX_ITERS; ++iter) {

      record.iter = iter + 1;

      problem.evaluate(results.xs_, results.us_, workspace.problem_data);
      problem.computeDerivatives(results.xs_, results.us_,
                                 workspace.problem_data);
      results.traj_cost_ =
          computeTrajectoryCost(problem, workspace.problem_data);
      computeInfeasibility(problem, results, workspace);
      record.prim_err = results.primal_infeasibility;

      backwardPass(problem, workspace, results);
      computeCriterion(workspace, results);
      record.dual_err = results.dual_infeasibility;
      record.merit = results.traj_cost_;
      record.inner_crit = 0.;

      if (results.dual_infeasibility < tol_) {
        results.conv = true;
        logger.log(record);
        break;
      }

      Scalar alpha_opt = 1;
      Scalar phi0 = results.traj_cost_;
      Scalar d1_phi, d2_phi;
      computeDirectionalDerivatives(workspace, results, d1_phi, d2_phi);
      // quadratic model lambda; captures by copy
      auto ls_model = [=, &problem, &workspace, &results](const Scalar alpha) {
        Scalar d1 = d1_phi;
        Scalar d2 = d2_phi;
        directionalDerivativeCorrection(problem, workspace, results, d1, d2);
        return phi0 + alpha * (d1 + 0.5 * d2 * alpha);
      };
      if (!(std::abs(d1_phi) < th_grad_)) {
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
      }
      forwardPass(problem, results, workspace, alpha_opt);
      results.xs_ = workspace.trial_xs_;
      results.us_ = workspace.trial_us_;

      record.dphi0 = d1_phi;
      record.step_size = alpha_opt;

      if (alpha_opt > th_step_dec_) {
        increase_reg();
      }
      if (alpha_opt <= th_step_inc_) {
        decrease_reg();
        if (xreg_ == reg_max_) {
          results.conv = false;
          break;
        }
      }

      invokeCallbacks(workspace, results);
      logger.log(record);
    }

    if (results.conv) {
      fmt::print(fmt::fg(fmt::color::dodger_blue), "Successfully converged.");
    } else {
      fmt::print(fmt::fg(fmt::color::red), "Convergence failure.");
    }
    fmt::print("\n");
    return results.conv;
  }

  static DynamicsDataTpl<Scalar> &
  stage_get_dynamics_data(StageDataTpl<Scalar> &sd) {
    return static_cast<DynamicsDataTpl<Scalar> &>(*sd.constraint_data[0]);
  }
};

} // namespace proxddp

#include "proxddp/fddp/solver-fddp.hxx"
