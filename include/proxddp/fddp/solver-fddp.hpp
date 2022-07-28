#pragma once

#include "proxddp/core/solver-base.hpp"
#include "proxddp/core/solver-results.hpp"
#include "proxddp/core/linesearch.hpp"
#include "proxddp/core/explicit-dynamics.hpp"

#include "proxddp/fddp/workspace.hpp"

#include "proxddp/utils/exceptions.hpp"
#include "proxddp/utils/logger.hpp"
#include "proxddp/utils/rollout.hpp"

#include <Eigen/Cholesky>

namespace proxddp {

template <typename Scalar> struct ResultsFDDP : ResultsBaseTpl<Scalar> {

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

  explicit ResultsFDDP(const TrajOptProblemTpl<Scalar> &problem);
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
  using Results = ResultsFDDP<Scalar>;
  using Workspace = WorkspaceFDDP<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using VParams = internal::value_storage<Scalar>;
  using QParams = internal::q_storage<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;
  using ExpModel = ExplicitDynamicsModelTpl<Scalar>;
  using ExpData = ExplicitDynamicsDataTpl<Scalar>;

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

  std::unique_ptr<Results> results_;
  std::unique_ptr<Workspace> workspace_;

  SolverFDDP(const Scalar tol = 1e-6, const Scalar reg_init = 1e-10,
             VerboseLevel verbose = VerboseLevel::QUIET);

  const Results &getResults() const { return *results_; }
  const Workspace &getWorkspace() const { return *workspace_; }

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
    const std::size_t nsteps = problem.numSteps();
    std::vector<VectorXs> &xs = workspace.trial_xs_;
    std::vector<VectorXs> &us = workspace.trial_us_;
    ProblemData &pd = workspace.problem_data;
    for (std::size_t i = 0; i < nsteps; i++) {
      auto ff = results.getFeedforward(i);
      auto fb = results.getFeedback(i);

      const StageModel &sm = *problem.stages_[i];
      StageData &sd = pd.getData(i);
      const DynamicsModelTpl<Scalar> &dm = sm.dyn_model();
      auto &dd = stage_get_dynamics_data(sd);
      // xref (-) xtrial
      const Manifold &space = sm.xspace();
      VectorXs &dx = workspace.dxs_[i];
      space.difference(xs[i], results.xs_[i], dx);
      us[i] = results.us_[i] + alpha * ff + fb * dx;
      forwardDynamics(space, dm, xs[i], us[i], dd, workspace.xnexts[i + 1]);

      space.integrate(workspace.xnexts[i + 1],
                      workspace.feas_gaps_[i + 1] * (alpha - 1.), xs[i + 1]);
    }
  }

  /// @brief Allocate workspace and results structs.
  void setup(const Problem &problem);

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
    for (i = nsteps - 1; i >= 0; i--) {

      const VParams &vnext = workspace.value_params[i + 1];
      QParams &qparam = workspace.q_params[i];

      StageModel &sm = *problem.stages_[i];
      const int nu = sm.nu();
      const int ndx1 = sm.ndx1();

      StageData &sd = prob_data.getData(i);
      const CostData &cd = *sd.cost_data;
      const ExpData &dd =
          static_cast<const ExpData &>(stage_get_dynamics_data(sd));

      /* Assemble Q-function */
      const ConstMatrixRef J_x_u(dd.jac_buffer_.leftCols(ndx1 + nu));

      qparam.q_2() = 2 * cd.value_;
      qparam.grad_ = cd.grad_;
      qparam.hess_ = cd.hess_;

      // TODO: implement second-order derivatives for the Q-function
      qparam.grad_.noalias() += J_x_u.transpose() * vnext.Vx_;
      qparam.hess_.noalias() += J_x_u.transpose() * vnext.Vxx_ * J_x_u;

      qparam.Quu_.diagonal().array() += ureg_;

      /* Compute gains */
      results.getFeedforward(i) = -qparam.Qu_;
      results.getFeedback(i) = -qparam.Qxu_.transpose();
      Eigen::LLT<MatrixXs> &llt = workspace.llts_[i];
      workspace.kkt_matrix_bufs[i] = qparam.Quu_;
      llt.compute(workspace.kkt_matrix_bufs[i]);
      llt.solveInPlace(results.gains_[i]);

      /* Compute value function */
      VParams &vcur = workspace.value_params[i];
      vcur.storage = qparam.storage.topLeftCorner(ndx1 + 1, ndx1 + 1) +
                     workspace.kkt_matrix_bufs[i] * results.gains_[i];
      vcur.Vxx_.diagonal().array() += xreg_;
      vcur.Vx_.noalias() += vcur.Vxx_ * workspace.feas_gaps_[i];
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

  bool run(const Problem &problem,
           const std::vector<VectorXs> &xs_init = DEFAULT_VECTOR<Scalar>,
           const std::vector<VectorXs> &us_init = DEFAULT_VECTOR<Scalar>) {
    Results &results = *results_;
    Workspace &workspace = *workspace_;

    checkTrajectoryAndAssign(problem, xs_init, us_init, results.xs_,
                             results.us_);

    ::proxddp::CustomLogger logger{};

    auto linesearch_fun = [&](const Scalar alpha) {
      tryStep(problem, results, workspace, alpha);
      return results.merit_value_;
    };

    std::size_t &iter = results.num_iters;
    for (iter = 0; iter < MAX_ITERS; ++iter) {

      LogRecord record;
      record.iter = iter;

      problem.evaluate(results.xs_, results.us_, workspace.problem_data);
      problem.computeDerivatives(results.xs_, results.us_,
                                 workspace.problem_data);
      results.traj_cost_ =
          computeTrajectoryCost(problem, workspace.problem_data);

      // TODO implement
      backwardPass(problem, workspace, results);

      // if something terminate
      if (false) {
        results.conv = true;
        break;
      }

      Scalar alpha_opt = 1;
      Scalar phi0 = results.traj_cost_;
      Scalar dphi0 = computeDirectionalDerivatives(problem, workspace);
      proxnlp::ArmijoLinesearch<Scalar>::run(
          linesearch_fun, phi0, dphi0, verbose_, ls_params.ls_beta,
          ls_params.armijo_c1, ls_params.alpha_min, alpha_opt);
      forwardPass(problem, results, workspace, alpha_opt);

      record.merit = phi0;
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

    return results.conv;
  }

  static DynamicsDataTpl<Scalar> &
  stage_get_dynamics_data(StageDataTpl<Scalar> &sd) {
    return static_cast<DynamicsDataTpl<Scalar> &>(*sd.constraint_data[0]);
  }
};

} // namespace proxddp

#include "proxddp/fddp/solver-fddp.hxx"
