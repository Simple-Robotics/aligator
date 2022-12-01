/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
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
  PROXDDP_NOMALLOC_BEGIN;
  const std::size_t nsteps = workspace.nsteps;
  std::vector<VectorXs> &xs_try = workspace.trial_xs;
  std::vector<VectorXs> &us_try = workspace.trial_us;
  const std::vector<VectorXs> &fs = workspace.dyn_slacks;
  ProblemData &prob_data = workspace.problem_data;

  {
    const auto &space = problem.stages_[0]->xspace_;
    workspace.dxs[0] = alpha * fs[0];
    space->integrate(results.xs[0], workspace.dxs[0], xs_try[0]);
  }
  Scalar traj_cost_ = 0.;

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &sm = *problem.stages_[i];
    StageData &sd = prob_data.getStageData(i);

    auto ff = results.getFeedforward(i);
    auto fb = results.getFeedback(i);

    workspace.dus[i] = alpha * ff;
    workspace.dus[i].noalias() += fb * workspace.dxs[i];
    sm.uspace().integrate(results.us[i], workspace.dus[i], us_try[i]);

    PROXDDP_NOMALLOC_END;
    sm.evaluate(xs_try[i], us_try[i], xs_try[i + 1], sd);
    PROXDDP_NOMALLOC_BEGIN;

    const ExpData &dd = stage_get_dynamics_data(sd);

    workspace.dxs[i + 1] = (alpha - 1.) * fs[i + 1]; // use as tmp variable
    sm.xspace_next_->integrate(dd.xnext_, workspace.dxs[i + 1], xs_try[i + 1]);
    const CostData &cd = *sd.cost_data;

    PROXDDP_RAISE_IF_NAN_NAME(xs_try[i + 1], fmt::format("xs[{}]", i + 1));
    PROXDDP_RAISE_IF_NAN_NAME(us_try[i], fmt::format("us[{}]", i));

    sm.xspace_->difference(results.xs[i + 1], xs_try[i + 1],
                           workspace.dxs[i + 1]);

    traj_cost_ += cd.value_;
  }
  CostData &cd_term = *prob_data.term_cost_data;

  PROXDDP_NOMALLOC_END;
  problem.term_cost_->evaluate(xs_try.back(), us_try.back(), cd_term);
  PROXDDP_NOMALLOC_BEGIN;

  traj_cost_ += cd_term.value_;
  const auto &space = problem.stages_.back()->xspace_next_;
  space->difference(results.xs[nsteps], xs_try[nsteps], workspace.dxs[nsteps]);

  prob_data.cost_ = traj_cost_;
  PROXDDP_NOMALLOC_END;
  return traj_cost_;
}

template <typename Scalar>
void SolverFDDP<Scalar>::expectedImprovement(Workspace &workspace, Scalar &d1,
                                             Scalar &d2) const {
  PROXDDP_NOMALLOC_BEGIN;
  Scalar &dg = workspace.dg_;
  Scalar &dq = workspace.dq_;
  Scalar &dv = workspace.dv_;
  dv = 0.;
  const std::size_t nsteps = workspace.nsteps;

  for (std::size_t i = 0; i <= nsteps; i++) {
    const VectorXs &ftVxx = workspace.ftVxx_[i];
    dv -= workspace.dxs[i].dot(ftVxx);
  }

  d1 = dg + dv;
  d2 = dq + 2 * dv;
  PROXDDP_NOMALLOC_END;
}

template <typename Scalar>
void SolverFDDP<Scalar>::updateExpectedImprovement(Workspace &workspace,
                                                   Results &results) const {
  PROXDDP_NOMALLOC_BEGIN;
  Scalar &dg = workspace.dg_;
  Scalar &dq = workspace.dq_;
  dg = 0.; // cost directional derivative
  dq = 0.; // cost 2nd direct. derivative
  const std::vector<VectorXs> &fs = workspace.dyn_slacks;
  const std::size_t nsteps = workspace.nsteps;

  // in croco: feedback/feedforward sign is flipped
  for (std::size_t i = 0; i < nsteps; i++) {
    const QParams &qparam = workspace.q_params[i];
    auto ff = results.getFeedforward(i);
    dg += qparam.Qu.dot(ff);
    dq += ff.dot(workspace.Quuks_[i]);
  }

  for (std::size_t i = 0; i <= nsteps; i++) {
    const VParams &vpar = workspace.value_params[i];
    dg += vpar.Vx_.dot(fs[i]);
    const VectorXs &ftVxx = workspace.ftVxx_[i];
    // ftVxx.noalias() = vpar.Vxx_ * fs[i];
    dq -= ftVxx.dot(fs[i]);
  }
  PROXDDP_NOMALLOC_END;
}

template <typename Scalar>
Scalar SolverFDDP<Scalar>::computeInfeasibility(const Problem &problem,
                                                const std::vector<VectorXs> &xs,
                                                Workspace &workspace) const {
  PROXDDP_NOMALLOC_BEGIN;
  const std::size_t nsteps = workspace.nsteps;
  const ProblemData &pd = workspace.problem_data;
  std::vector<VectorXs> &fs = workspace.dyn_slacks;

  const VectorXs &x0 = problem.getInitState();
  const auto &space = problem.stages_[0]->xspace_;
  space->difference(xs[0], x0, fs[0]);

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &sm = *problem.stages_[i];
    const auto &sd = pd.getStageData(i);
    const ExpData &dd = stage_get_dynamics_data(sd);
    sm.xspace_->difference(xs[i + 1], dd.xnext_, fs[i + 1]);
  }
  Scalar res = math::infty_norm(fs);
  PROXDDP_NOMALLOC_END;
  return res;
}

template <typename Scalar>
Scalar SolverFDDP<Scalar>::computeCriterion(Workspace &workspace) {
  PROXDDP_NOMALLOC_BEGIN;
  const std::size_t nsteps = workspace.nsteps;
  Scalar v = 0.;
  for (std::size_t i = 0; i < nsteps; i++) {
    Scalar s = math::infty_norm(workspace.q_params[i].Qu);
    v = std::max(v, s);
  }
  PROXDDP_NOMALLOC_END;
  return v;
}

template <typename Scalar>
void SolverFDDP<Scalar>::backwardPass(const Problem &problem,
                                      Workspace &workspace,
                                      Results &results) const {
  PROXDDP_NOMALLOC_BEGIN;

  const std::size_t nsteps = workspace.nsteps;
  const std::vector<VectorXs> &fs = workspace.dyn_slacks;

  ProblemData &prob_data = workspace.problem_data;
  {
    const CostData &term_cost_data = *prob_data.term_cost_data;
    VParams &vp = workspace.value_params[nsteps];
    vp.v_ = term_cost_data.value_;
    vp.Vx_ = term_cost_data.Lx_;
    vp.Vxx_ = term_cost_data.Lxx_;
    vp.Vxx_.diagonal().array() += xreg_;
    VectorXs &ftVxx = workspace.ftVxx_[nsteps];
    ftVxx.noalias() = vp.Vxx_ * fs[nsteps];
    vp.Vx_ += ftVxx;
  }

  for (std::size_t i = nsteps; i-- > 0;) {
    const VParams &vnext = workspace.value_params[i + 1];
    QParams &qparam = workspace.q_params[i];

    StageModel &sm = *problem.stages_[i];
    StageData &sd = prob_data.getStageData(i);

    const int nu = sm.nu();
    const int ndx1 = sm.ndx1();
    assert(qparam.grad_.size() == ndx1 + nu);

    const CostData &cd = *sd.cost_data;
    const DynamicsDataTpl<Scalar> &dd = sd.dyn_data();

    /* Assemble Q-function */
    auto J_x_u = dd.jac_buffer_.leftCols(ndx1 + nu);

    qparam.q_ = cd.value_;
    qparam.grad_ = cd.grad_;
    qparam.grad_.noalias() += J_x_u.transpose() * vnext.Vx_;

    // TODO: implement second-order derivatives for the Q-function
    workspace.JtH_temp_[i].noalias() = J_x_u.transpose() * vnext.Vxx_;
    qparam.hess_ = cd.hess_;
    qparam.hess_.noalias() += workspace.JtH_temp_[i] * J_x_u;
    qparam.Quu.diagonal().array() += ureg_;

    /* Compute gains */

    auto ff = results.getFeedforward(i);
    auto fb = results.getFeedback(i);
    ff = -qparam.Qu;
    fb = -qparam.Qxu.transpose();

    Eigen::LLT<MatrixXs> &llt = workspace.llts_[i];
    llt.compute(qparam.Quu);
    llt.solveInPlace(results.gains_[i]);

#ifndef NDEBUG
    {
      PROXDDP_NOMALLOC_END;
      std::FILE *fi = std::fopen("fddp.log", "a");
      if (i == workspace.nsteps - 1)
        fmt::print(fi, "[backward {:d}]\n", results.num_iters + 1);
      fmt::print(fi, "uff[{:d}]={}\n", i, ff.head(nu).transpose());
      fmt::print(fi, "V'x[{:d}]={}\n", i, vnext.Vx_.transpose());
      std::fclose(fi);
      PROXDDP_NOMALLOC_BEGIN;
    }
#endif
    workspace.Quuks_[i].noalias() = qparam.Quu * ff;

    /* Compute value function */
    VParams &vp = workspace.value_params[i];
    vp.Vx_ = qparam.Qx;
    vp.Vx_.noalias() += fb.transpose() * qparam.Qu;
    vp.Vxx_ = qparam.Qxx;
    vp.Vxx_.noalias() += qparam.Qxu * fb;
    vp.Vxx_ = vp.Vxx_.template selfadjointView<Eigen::Lower>();
    vp.Vxx_.diagonal().array() += xreg_;
    VectorXs &ftVxx = workspace.ftVxx_[i];
    ftVxx.noalias() = vp.Vxx_ * fs[i];
    vp.Vx_ += ftVxx;
  }

  PROXDDP_NOMALLOC_END;
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
  results.xs[0] = problem.getInitState();
  results.conv = false;

  logger.active = verbose_ > 0;
  logger.start();

  // in Crocoddyl, linesearch xs is primed to use problem x0

  const auto linesearch_fun = [&](const Scalar alpha) {
    return forwardPass(problem, results, workspace, alpha);
  };

  Scalar d1_phi = 0., d2_phi = 0.;
  Scalar phi0;
  // linesearch model oracle
  const auto ls_model = [&](const Scalar alpha) {
    expectedImprovement(workspace, d1_phi, d2_phi);
    return phi0 + alpha * (d1_phi + 0.5 * d2_phi * alpha);
  };

  LogRecord record;

  std::size_t &iter = results.num_iters;
  for (iter = 0; iter <= max_iters; ++iter) {

    record.iter = iter + 1;

    if (iter == 0) {
      results.traj_cost_ =
          problem.evaluate(results.xs, results.us, workspace.problem_data);
    }
    problem.computeDerivatives(results.xs, results.us, workspace.problem_data);
    results.prim_infeas = computeInfeasibility(problem, results.xs, workspace);
    PROXDDP_RAISE_IF_NAN(results.prim_infeas);
    record.prim_err = results.prim_infeas;

    backwardPass(problem, workspace, results);
    results.dual_infeas = computeCriterion(workspace);
    PROXDDP_RAISE_IF_NAN(results.dual_infeas);
    record.dual_err = results.dual_infeas;

    Scalar stopping_criterion =
        std::max(results.prim_infeas, results.dual_infeas);
    if (stopping_criterion < target_tol_) {
      results.conv = true;
      break;
    }

    if (iter == max_iters)
      break;

    phi0 = results.traj_cost_;
    PROXDDP_RAISE_IF_NAN(phi0);

    updateExpectedImprovement(workspace, results);

    Scalar alpha_opt;
    Scalar phi_new = FDDPGoldsteinLinesearch<Scalar>::run(
        linesearch_fun, ls_model, phi0, ls_params, th_grad_, d1_phi, alpha_opt);

    results.traj_cost_ = phi_new;
    PROXDDP_RAISE_IF_NAN(alpha_opt);
    PROXDDP_RAISE_IF_NAN(phi_new);
    record.step_size = alpha_opt;
    record.dM = phi_new - phi0;
    record.merit = phi_new;
    record.dphi0 = d1_phi;
    record.xreg = xreg_;

    results.xs = workspace.trial_xs;
    results.us = workspace.trial_us;
    if (std::abs(d1_phi) < th_grad_) {
      results.conv = true;
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

  if (iter < max_iters)
    logger.log(record);
  logger.finish(results.conv);
  return results.conv;
}
} // namespace proxddp
