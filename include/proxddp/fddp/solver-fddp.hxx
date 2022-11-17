#pragma once

#include "proxddp/fddp/solver-fddp.hpp"

namespace proxddp {

/* SolverFDDP<Scalar> */

template <typename Scalar>
SolverFDDP<Scalar>::SolverFDDP(const Scalar tol, VerboseLevel verbose,
                               const Scalar reg_init,
                               const std::size_t max_iters)
    : target_tol_(tol), reg_init(reg_init), verbose_(verbose),
      max_iters(max_iters) {}

template <typename Scalar>
void SolverFDDP<Scalar>::setup(const Problem &problem) {
  results_ = std::make_unique<Results>(problem);
  workspace_ = std::make_unique<Workspace>(problem);
  // check if there are any constraints other than dynamics and throw a warning
  std::vector<std::size_t> idx_where_constraints;
  for (std::size_t i = 0; i < problem.numSteps(); i++) {
    const shared_ptr<StageModel> &sm = problem.stages_[i];
    if (sm->numConstraints() > 1) {
      idx_where_constraints.push_back(i);
    }
  }
  if (idx_where_constraints.size() > 0) {
    PROXDDP_FDDP_WARNING(
        fmt::format("problem stages [{}] have constraints, "
                    "which this solver cannot handle. "
                    "Please use a penalized cost formulation.\n",
                    fmt::join(idx_where_constraints, ", ")));
  }
  if (problem.term_constraint_) {
    PROXDDP_FDDP_WARNING(
        "problem has a terminal constraint, which this solver cannot "
        "handle.\n");
  }
}

template <typename Scalar>
Scalar
SolverFDDP<Scalar>::forwardPass(const Problem &problem, const Results &results,
                                Workspace &workspace, const Scalar alpha) {
  const std::size_t nsteps = workspace.nsteps;
  std::vector<VectorXs> &xs_try = workspace.trial_xs;
  std::vector<VectorXs> &us_try = workspace.trial_us;
  std::vector<VectorXs> &xnexts = workspace.xnexts_;
  const std::vector<VectorXs> &fs = workspace.dyn_slacks;
  ProblemData &pd = workspace.problem_data;

  {
    const Manifold &space = problem.stages_[0]->xspace();
    space.integrate(results.xs[0], alpha * fs[0], xs_try[0]);
  }
  Scalar traj_cost_ = 0.;

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &sm = *problem.stages_[i];
    StageData &sd = pd.getStageData(i);

    ConstVectorRef ff = results.getFeedforward(i);
    ConstMatrixRef fb = results.getFeedback(i);

    // WARNING: this is the reverse of crocoddyl
    // which has its gains' sign flipped
    sm.xspace().difference(results.xs[i], xs_try[i], workspace.dxs[i]);
    sm.uspace().integrate(results.us[i], alpha * ff + fb * workspace.dxs[i],
                          us_try[i]);
    sm.evaluate(xs_try[i], us_try[i], xs_try[i + 1], sd);
    const ExpData &dd = stage_get_dynamics_data(sd);
    xnexts[i + 1] = dd.xnext_;
    sm.xspace_next().integrate(xnexts[i + 1], fs[i + 1] * (alpha - 1.),
                               xs_try[i + 1]);
    const CostData &cd = *sd.cost_data;

    PROXDDP_RAISE_IF_NAN_NAME(xs_try[i + 1], fmt::format("xs[{}]", i + 1));
    PROXDDP_RAISE_IF_NAN_NAME(us_try[i], fmt::format("us[{}]", i));

    traj_cost_ += cd.value_;
  }
  CostData &cd_term = *pd.term_cost_data;
  problem.term_cost_->evaluate(xs_try.back(), us_try.back(), cd_term);
  traj_cost_ += cd_term.value_;
  const Manifold &space = problem.stages_.back()->xspace();
  space.difference(results.xs[nsteps], xs_try[nsteps], workspace.dxs[nsteps]);
  return traj_cost_;
}

template <typename Scalar>
void SolverFDDP<Scalar>::expectedImprovement(Workspace &workspace, Scalar &d1,
                                             Scalar &d2) const {
  // equivalent to expectedImprovement() in crocoddyl
  Scalar &dg_ = workspace.dg_;
  Scalar &dq_ = workspace.dq_;
  Scalar &dv_ = workspace.dv_;
  dv_ = 0.;
  const std::size_t nsteps = workspace.nsteps;
  const std::vector<VectorXs> &fs = workspace.dyn_slacks;

  for (std::size_t i = 0; i <= nsteps; i++) {
    const VParams &vp = workspace.value_params[i];
    VectorXs &ftVxx = workspace.ftVxx_[i];
    ftVxx.noalias() = vp.Vxx() * workspace.dxs[i];
    dv_ -= fs[i].dot(ftVxx);
  }

  d1 = dg_ + dv_;
  d2 = dq_ + 2 * dv_;
}

template <typename Scalar>
void SolverFDDP<Scalar>::updateExpectedImprovement(Workspace &workspace,
                                                   Results &results) const {
  // equivalent to updateExpectedImprovement() in crocoddyl
  Scalar &dg_ = workspace.dg_;
  Scalar &dq_ = workspace.dq_;
  dg_ = 0.; // cost directional derivative
  dq_ = 0.; // cost 2nd direct. derivative
  const std::vector<VectorXs> &fs = workspace.dyn_slacks;
  const std::size_t nsteps = workspace.nsteps;

  // in croco: feedback/feedforward sign is flipped
  for (std::size_t i = 0; i <= nsteps; i++) {
    if (i < nsteps) {
      const QParams &qpar = workspace.q_params[i];
      ConstVectorRef ff = results.getFeedforward(i);
      dg_ += qpar.Qu.dot(ff);
      dq_ += ff.dot(workspace.Quuks_[i]);
    }
    const VParams &vpar = workspace.value_params[i];
    dg_ += vpar.Vx().dot(fs[i]);
    VectorXs &ftVxx = workspace.ftVxx_[i];
    ftVxx.noalias() = vpar.Vxx() * fs[i];
    dq_ -= ftVxx.dot(fs[i]);
  }
}

template <typename Scalar>
Scalar SolverFDDP<Scalar>::computeInfeasibility(const Problem &problem,
                                                const std::vector<VectorXs> &xs,
                                                Workspace &workspace) {
  const std::size_t nsteps = workspace.nsteps;
  const ProblemData &pd = workspace.problem_data;
  std::vector<VectorXs> &xnexts = workspace.xnexts_;
  std::vector<VectorXs> &fs = workspace.dyn_slacks;

  const VectorXs &x0 = problem.getInitState();
  const Manifold &space = problem.stages_[0]->xspace();
  space.difference(xs[0], x0, fs[0]);

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &sm = *problem.stages_[i];
    const ExpData &dd = stage_get_dynamics_data(pd.getStageData(i));
    xnexts[i + 1] = dd.xnext_;
    sm.xspace().difference(xs[i + 1], xnexts[i + 1], fs[i + 1]);
  }
  return math::infty_norm(workspace.dyn_slacks);
}

template <typename Scalar>
Scalar SolverFDDP<Scalar>::computeCriterion(Workspace &workspace) {
  const std::size_t nsteps = workspace.nsteps;
  std::vector<ConstVectorRef> Qus;
  for (std::size_t i = 0; i < nsteps; i++) {
    Qus.push_back(workspace.q_params[i].Qu);
#ifndef NDEBUG
    std::FILE *fi = std::fopen("fddp.log", "a");
    fmt::print(fi, "Qu[{:d}]={:.3e}\n", i, math::infty_norm(Qus.back()));
    std::fclose(fi);
#endif
  }
  return math::infty_norm(Qus);
}

template <typename Scalar>
void SolverFDDP<Scalar>::backwardPass(const Problem &problem,
                                      Workspace &workspace,
                                      Results &results) const {

  const std::size_t nsteps = workspace.nsteps;
  const std::vector<VectorXs> &fs = workspace.dyn_slacks;

  ProblemData &prob_data = workspace.problem_data;
  {
    const CostData &term_cost_data = *prob_data.term_cost_data;
    VParams &vp = workspace.value_params[nsteps];
    vp.v_2() = 2 * term_cost_data.value_;
    vp.Vx() = term_cost_data.Lx_;
    vp.Vxx() = term_cost_data.Lxx_;
    vp.Vxx().diagonal().array() += xreg_;
    VectorXs &ftVxx = workspace.ftVxx_[nsteps];
    vp.Vx() += vp.Vxx() * fs[nsteps];
    vp.storage = vp.storage.template selfadjointView<Eigen::Lower>();
  }

  std::size_t i;
  for (std::size_t k = 0; k < nsteps; k++) {
    i = nsteps - k - 1;
    const VParams &vnext = workspace.value_params[i + 1];
    QParams &qparam = workspace.q_params[i];

    StageModel &sm = *problem.stages_[i];
    StageData &sd = prob_data.getStageData(i);

    const int nu = sm.nu();
    const int ndx1 = sm.ndx1();
    assert((qparam.storage.cols() == ndx1 + nu + 1) &&
           (qparam.storage.rows() == ndx1 + nu + 1));
    assert(qparam.grad_.size() == ndx1 + nu);

    const CostData &cd = *sd.cost_data;
    DynamicsDataTpl<Scalar> &dd = sd.dyn_data();

    /* Assemble Q-function */
    ConstMatrixRef J_x_u = dd.jac_buffer_.leftCols(ndx1 + nu);

    qparam.q_2() = 2 * cd.value_;
    qparam.grad_ = cd.grad_;
    qparam.hess_ = cd.hess_;

    // TODO: implement second-order derivatives for the Q-function
    qparam.grad_.noalias() += J_x_u.transpose() * vnext.Vx();
    qparam.hess_.noalias() += J_x_u.transpose() * vnext.Vxx() * J_x_u;
    qparam.Quu.diagonal().array() += ureg_;
    qparam.storage = qparam.storage.template selfadjointView<Eigen::Lower>();

    /* Compute gains */
    // MatrixXs &kkt_mat = workspace.kkt_mat_bufs[i];
    MatrixXs &kkt_rhs = workspace.kkt_rhs_bufs[i];

    // kkt_mat = qparam.Quu;
    VectorRef ffwd = results.getFeedforward(i);
    MatrixRef fback = results.getFeedback(i);
    ffwd = -qparam.Qu;
    fback = -qparam.Qxu.transpose();
    kkt_rhs = results.gains_[i];

    Eigen::LLT<MatrixXs> &llt = workspace.llts_[i];
    llt.compute(qparam.Quu);
    llt.solveInPlace(results.gains_[i]);

    workspace.Quuks_[i] = qparam.Quu * ffwd;
#ifndef NDEBUG
    auto ff = results.getFeedforward(i);
    std::FILE *fi = std::fopen("fddp.log", "a");
    if (i == workspace.nsteps - 1)
      fmt::print(fi, "[backward {:d}]\n", results.num_iters + 1);
    fmt::print(fi, "uff[{:d}]={}\n", i, ff.head(nu).transpose());
    fmt::print(fi, "V'x[{:d}]={}\n", i, vnext.Vx().transpose());
    std::fclose(fi);
#endif

    /* Compute value function */
    VParams &vp = workspace.value_params[i];
    vp.Vx() = qparam.Qx + fback.transpose() * qparam.Qu;
    vp.Vxx() = qparam.Qxx + qparam.Qxu * fback;
    vp.Vxx().diagonal().array() += xreg_;
    vp.Vx() += vp.Vxx() * fs[i];
    vp.storage = vp.storage.template selfadjointView<Eigen::Lower>();
  }
}

template <typename Scalar>
bool SolverFDDP<Scalar>::run(const Problem &problem,
                             const std::vector<VectorXs> &xs_init,
                             const std::vector<VectorXs> &us_init) {
  xreg_ = reg_init;
  ureg_ = xreg_;

#ifndef NDEBUG
  std::FILE *fi = std::fopen("fddp.log", "w");
  std::fclose(fi);
#endif

  if (results_ == 0 || workspace_ == 0) {
    PROXDDP_RUNTIME_ERROR(
        "Either results or workspace not allocated. Call setup() first!");
  }
  Results &results = *results_;
  Workspace &workspace = *workspace_;

  checkTrajectoryAndAssign(problem, xs_init, us_init, results.xs, results.us);

  logger.active = verbose_ > 0;
  logger.start();

  // in Crocoddyl, linesearch xs is primed to use problem x0
  workspace.xnexts_[0] = problem.getInitState();

  auto linesearch_fun = [&](const Scalar alpha) {
    return forwardPass(problem, results, workspace, alpha);
  };

  LogRecord record;
  record.inner_crit = 0.;
  record.dual_err = 0.;
  record.dphi0 = 0.;
  std::size_t &iter = results.num_iters;
  for (iter = 0; iter <= max_iters; ++iter) {

    record.iter = iter + 1;

    problem.evaluate(results.xs, results.us, workspace.problem_data);
    results.traj_cost_ = computeTrajectoryCost(problem, workspace.problem_data);
    results.prim_infeas = computeInfeasibility(problem, results.xs, workspace);
    problem.computeDerivatives(results.xs, results.us, workspace.problem_data);

    backwardPass(problem, workspace, results);
    results.dual_infeas = computeCriterion(workspace);

    PROXDDP_RAISE_IF_NAN(results.prim_infeas);
    PROXDDP_RAISE_IF_NAN(results.dual_infeas);
    record.prim_err = results.prim_infeas;
    record.dual_err = results.dual_infeas;
    record.merit = results.traj_cost_;
    record.inner_crit = 0.;
    record.xreg = xreg_;

    Scalar stopping_criterion =
        std::max(results.prim_infeas, results.dual_infeas);
    if (stopping_criterion < target_tol_) {
      results.conv = true;
      break;
    }

    if (iter >= max_iters) {
      break;
    }

    Scalar phi0 = results.traj_cost_;
    PROXDDP_RAISE_IF_NAN(phi0);
    Scalar d1_phi = 0., d2_phi = 0.;
    updateExpectedImprovement(workspace, results);
    PROXDDP_RAISE_IF_NAN(d1_phi);
    PROXDDP_RAISE_IF_NAN(d2_phi);

    // quadratic model lambda; captures by copy
    auto ls_model = [&](Scalar alpha) {
      expectedImprovement(workspace, d1_phi, d2_phi);
      return phi0 + alpha * (d1_phi + 0.5 * d2_phi * alpha);
    };

    Scalar alpha_opt = 1;
    alpha_opt = FDDPGoldsteinLinesearch<Scalar>::run(
        linesearch_fun, ls_model, phi0, ls_params, d1_phi, th_grad_);
    record.step_size = alpha_opt;
    Scalar phi_new = linesearch_fun(alpha_opt);
    PROXDDP_RAISE_IF_NAN(phi_new);
    record.merit = phi_new;
    record.dM = phi_new - phi0;
    record.dphi0 = d1_phi;

    results.xs = workspace.trial_xs;
    results.us = workspace.trial_us;
    if (std::abs(d1_phi) < th_grad_) {
      results.conv = true;
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
    logger.log(record);
  }

  logger.finish(results.conv);
  return results.conv;
}
} // namespace proxddp
