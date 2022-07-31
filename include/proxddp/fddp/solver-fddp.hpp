#pragma once

#include "proxddp/core/solver-base.hpp"
#include "proxddp/core/solver-results.hpp"
#include "proxddp/core/linesearch.hpp"
#include "proxddp/core/helpers-base.hpp"
#include "proxddp/core/explicit-dynamics.hpp"

#include "proxddp/fddp/workspace.hpp"

#include "proxddp/utils/exceptions.hpp"
#include "proxddp/utils/logger.hpp"
#include "proxddp/utils/rollout.hpp"

#include <fmt/ostream.h>

#include <Eigen/Cholesky>

#define proxddp_fddp_warning(msg)                                              \
  fmt::print(fmt::fg(fmt::color::yellow), "[SolverFDDP] ({}) warning: {}",     \
             __FUNCTION__, msg)

namespace proxddp {

template <typename Scalar> struct ResultsFDDPTpl : ResultsBaseTpl<Scalar> {

  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ResultsBaseTpl<Scalar>;
  using BlockXs = Eigen::Block<MatrixXs, -1, -1>;

  using Base::gains_;
  using Base::us_;
  using Base::xs_;

  decltype(auto) getFeedforward(std::size_t i) { return gains_[i].col(0); }
  decltype(auto) getFeedforward(std::size_t i) const {
    return gains_[i].col(0);
  }

  decltype(auto) getFeedback(std::size_t i) {
    const int ndx = this->gains_[i].cols() - 1;
    return gains_[i].rightCols(ndx);
  }

  decltype(auto) getFeedback(std::size_t i) const {
    const int ndx = this->gains_[i].cols() - 1;
    return gains_[i].rightCols(ndx);
  }

  explicit ResultsFDDPTpl(const TrajOptProblemTpl<Scalar> &problem);
};

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

  VerboseLevel verbose_;
  /// Maximum number of iterations for the solver.
  std::size_t MAX_ITERS = 200;

  /// Callbacks
  std::vector<CallbackPtr> callbacks_;

  std::unique_ptr<Results> results_;
  std::unique_ptr<Workspace> workspace_;

  SolverFDDP(const Scalar tol = 1e-6, const Scalar reg_init = 1e-10,
             VerboseLevel verbose = VerboseLevel::QUIET);

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

  Scalar computeDirectionalDerivatives(const Problem &problem,
                                       const Workspace &workspace) const {
    return 0.;
  }

  /// @brief  Perform a nonlinear rollout with an additional defect tacked onto
  /// the state trajectory.
  void forwardPass(const Problem &problem, const Results &results,
                   Workspace &workspace, const Scalar alpha) const {
    const std::size_t nsteps = workspace.nsteps;
    std::vector<VectorXs> &xs_try = workspace.trial_xs_;
    std::vector<VectorXs> &us_try = workspace.trial_us_;
    ProblemData &pd = workspace.problem_data;

    for (std::size_t i = 0; i < nsteps; i++) {
      const StageModel &sm = *problem.stages_[i];
      const DynamicsModelTpl<Scalar> &dm = sm.dyn_model();
      const Manifold &space = sm.xspace();
      StageData &sd = pd.getData(i);
      DynamicsDataTpl<Scalar> &dd = stage_get_dynamics_data(sd);

      VectorXs &dx = workspace.dxs_[i];
      ConstVectorRef ff = results.getFeedforward(i);
      ConstMatrixRef fb = results.getFeedback(i);

      space.difference(results.xs_[i], xs_try[i], dx);
      us_try[i] = results.us_[i] + alpha * ff + fb * dx;
      forwardDynamics(space, dm, xs_try[i], us_try[i], dd,
                      workspace.xnexts_[i]);

      space.integrate(workspace.xnexts_[i],
                      workspace.feas_gaps_[i + 1] * (alpha - 1.),
                      xs_try[i + 1]);
    }
  }

  /**
   * @brief   Computes dynamical feasibility gaps.
   * @details This computes the difference \f$x_{i+1} \ominus \f$, as well as
   *          the residual of initial condition.
   *
   */
  void computeInfeasibility(const Problem &problem, Results &results,
                            Workspace &workspace) const;

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

    ::proxddp::CustomLogger logger{};
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
      Scalar dphi0 = computeDirectionalDerivatives(problem, workspace);
      proxnlp::ArmijoLinesearch<Scalar>::run(
          linesearch_fun, phi0, dphi0, verbose_, ls_params.ls_beta,
          ls_params.armijo_c1, ls_params.alpha_min, alpha_opt);
      results.xs_ = workspace.trial_xs_;
      results.us_ = workspace.trial_us_;

      record.dphi0 = dphi0;
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

      logger.log(record);
    }

    if (results.conv) {
      fmt::print(fmt::fg(fmt::color::dodger_blue), "Successfully converged.\n");
    } else {
      fmt::print(fmt::fg(fmt::color::red), "Convergence failure.\n");
    }
    return results.conv;
  }

  static DynamicsDataTpl<Scalar> &
  stage_get_dynamics_data(StageDataTpl<Scalar> &sd) {
    return static_cast<DynamicsDataTpl<Scalar> &>(*sd.constraint_data[0]);
  }
};

} // namespace proxddp

#include "proxddp/fddp/solver-fddp.hxx"
