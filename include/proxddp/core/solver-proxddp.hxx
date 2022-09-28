/// @file solver-proxddp.hxx
/// @brief  Implementations for the trajectory optimization algorithm.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include <fmt/color.h>

namespace proxddp {

#ifndef NDEBUG
const char *LS_DEBUG_LOG_PATH = "assets/linesearch_iter.txt";
#endif

template <typename Scalar>
SolverProxDDP<Scalar>::SolverProxDDP(const Scalar tol, const Scalar mu_init,
                                     const Scalar rho_init,
                                     const std::size_t max_iters,
                                     const VerboseLevel verbose)
    : target_tol_(tol), mu_init(mu_init), rho_init(rho_init), verbose_(verbose),
      MAX_ITERS(max_iters), merit_fun(this) {
  ls_params.alpha_min = 1e-7;
  ls_params.interp_type = proxnlp::LSInterpolation::CUBIC;
  if (mu_init >= 1.) {
    proxddp_runtime_error(
        fmt::format("Penalty value mu_init={:g}>=1!", mu_init));
  }
}

template <typename Scalar>
void SolverProxDDP<Scalar>::linearRollout(const Problem &problem,
                                          Workspace &workspace,
                                          const Results &results) const {
  computeDirX0(problem, workspace, results);

  const std::size_t nsteps = workspace.nsteps;

  for (std::size_t i = 0; i < nsteps; i++) {
    VectorXs &pd_step = workspace.pd_step_[i + 1];
    const auto ff = results.getFeedforward(i);
    const auto fb = results.getFeedback(i);

    pd_step = ff + fb * workspace.dxs_[i];
  }
  if (problem.term_constraint_) {
    const auto ff = results.getFeedforward(nsteps);
    const auto fb = results.getFeedback(nsteps);
    workspace.dlams_.back() = ff + fb * workspace.dxs_[nsteps];
  }
}

template <typename Scalar>
void SolverProxDDP<Scalar>::tryStep(const Problem &problem,
                                    Workspace &workspace,
                                    const Results &results,
                                    const Scalar alpha) const {

  const std::size_t nsteps = problem.numSteps();

  for (std::size_t i = 0; i <= nsteps; i++)
    workspace.trial_lams[i] = results.lams[i] + alpha * workspace.dlams_[i];

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];
    stage.xspace_->integrate(results.xs[i], alpha * workspace.dxs_[i],
                             workspace.trial_xs[i]);
    stage.uspace_->integrate(results.us[i], alpha * workspace.dus_[i],
                             workspace.trial_us[i]);
  }
  const StageModel &stage = *problem.stages_[nsteps - 1];
  stage.xspace_next_->integrate(results.xs[nsteps],
                                alpha * workspace.dxs_[nsteps],
                                workspace.trial_xs[nsteps]);

  problem.evaluate(workspace.trial_xs, workspace.trial_us,
                   workspace.trial_prob_data);
}

template <typename Scalar>
void SolverProxDDP<Scalar>::computeDirX0(const Problem &problem,
                                         Workspace &workspace,
                                         const Results &results) const {
  // compute direction dx0
  const VParams &vp = workspace.value_params[0];
  const FunctionData &init_data = *workspace.problem_data.init_data;
  const int ndual0 = problem.init_state_error.nr;
  const int ndx0 = problem.init_state_error.ndx1;
  const VectorXs &lampl0 = workspace.lams_plus[0];
  const VectorXs &lamin0 = results.lams[0];
  // const VectorXs &prevlam0 = workspace.prev_lams[0];
  const CostData &proxdata0 = *workspace.prox_datas[0];
  MatrixXs &kkt_mat = workspace.kkt_mat_buf_[0];
  VectorRef kkt_rhs_0 = workspace.kkt_rhs_buf_[0].col(0);

  auto kktx = kkt_rhs_0.head(ndx0);
  auto kktl = kkt_rhs_0.tail(ndual0);
  kktx = vp.Vx() + init_data.Jx_.transpose() * lamin0 + rho() * proxdata0.Lx_;
  kktl = mu() * (lampl0 - lamin0);
  {
    workspace.pd_step_[0].setZero();
    workspace.trial_xs[0] = problem.getInitState();
    workspace.trial_lams[0].setZero();
    kkt_rhs_0.setZero();
    workspace.dual_infeas_by_stage(0) = 0.;
    return;
  }

  kkt_mat.setZero();
  kkt_mat.topLeftCorner(ndx0, ndx0) = vp.Vxx() + rho() * proxdata0.Lxx_;
  // kkt_mat.topLeftCorner(ndx0, ndx0) += init_data.Hxx_;
  kkt_mat.topRightCorner(ndx0, ndual0) = init_data.Jx_.transpose();
  kkt_mat.bottomLeftCorner(ndual0, ndx0) = init_data.Jx_;
  kkt_mat.bottomRightCorner(ndual0, ndual0).diagonal().array() = -mu();
  Eigen::LDLT<MatrixXs, Eigen::Lower> &ldlt = workspace.ldlts_[0];
  ldlt.compute(kkt_mat);
  assert(workspace.pd_step_[0].size() == kkt_rhs_0.size());
  workspace.pd_step_[0] = -kkt_rhs_0;
  ldlt.solveInPlace(workspace.pd_step_[0]);
  const ProxData &proxdata = *workspace.prox_datas[0];
  workspace.inner_criterion_x = math::infty_norm(kktx);
  workspace.inner_criterion_u = 0.;
  workspace.inner_criterion_l = math::infty_norm(kktl);
  workspace.inner_criterion_by_stage(0) = math::infty_norm(kkt_rhs_0);
  workspace.dual_infeas_by_stage(0) =
      math::infty_norm(kktx - rho() * proxdata.Lx_);
}

template <typename Scalar>
void SolverProxDDP<Scalar>::setup(const Problem &problem) {
  workspace_ = std::make_unique<Workspace>(problem);
  results_ = std::make_unique<Results>(problem);

  Workspace &ws = *workspace_;
  prox_penalties_.clear();
  const std::size_t nsteps = problem.numSteps();
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &sm = *problem.stages_[i];
    prox_penalties_.emplace_back(sm.xspace_, sm.uspace_, ws.prev_xs[i],
                                 ws.prev_us[i], false);
    if (i == nsteps - 1) {
      prox_penalties_.emplace_back(sm.xspace_next_, sm.uspace_,
                                   ws.prev_xs[nsteps], problem.dummy_term_u0,
                                   true);
    }
  }

  for (std::size_t i = 0; i < nsteps + 1; i++) {
    const ProxPenaltyType *penal = &prox_penalties_[i];
    ws.prox_datas.push_back(std::make_shared<ProxData>(penal));
  }

  assert(prox_penalties_.size() == (nsteps + 1));
  assert(ws.prox_datas.size() == (nsteps + 1));
}

template <typename Scalar>
bool SolverProxDDP<Scalar>::backwardPass(const Problem &problem,
                                         Workspace &workspace,
                                         Results &results) const {
  /* Terminal node */
  computeTerminalValue(problem, workspace, results);

  const std::size_t nsteps = problem.numSteps();
  for (std::size_t i = 0; i < nsteps; i++) {
    std::size_t t = nsteps - i - 1;
    updateHamiltonian(problem, t, results, workspace);
    bool b = computeGains(problem, workspace, results, t);
    if (!b) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
void SolverProxDDP<Scalar>::computeTerminalValue(const Problem &problem,
                                                 Workspace &workspace,
                                                 Results &results) const {
  const std::size_t nsteps = problem.numSteps();

  const TrajOptDataTpl<Scalar> &prob_data = workspace.problem_data;
  const CostData &term_cost_data = *prob_data.term_cost_data;
  VParams &term_value = workspace.value_params[nsteps];
  const CostData &proxdata = *workspace.prox_datas[nsteps];

  term_value.v_2() = 2 * (term_cost_data.value_ + rho() * proxdata.value_);
  term_value.Vx() = term_cost_data.Lx_ + rho() * proxdata.Lx_;
  term_value.Vxx() = term_cost_data.Lxx_ + rho() * proxdata.Lxx_;

  if (problem.term_constraint_) {
    /* check number of multipliers */
    assert(results.lams.size() == (nsteps + 2));
    assert(results.gains_.size() == (nsteps + 1));
    const Constraint &term_cstr = *problem.term_constraint_;
    const FunctionData &cstr_data = *prob_data.term_cstr_data;

    VectorXs &lamplus = workspace.lams_plus[nsteps + 1];
    // const VectorXs &lamprev = workspace.prev_lams[nsteps + 1];
    const VectorXs &lamin = results.lams[nsteps + 1];
    const MatrixRef &cJx = cstr_data.Jx_;

    auto ff = results.getFeedforward(nsteps);
    auto fb = results.getFeedback(nsteps);
    ff = lamplus - lamin;
    fb = mu_inv() * cJx;

    term_value.v_2() += mu_inv() * lamplus.squaredNorm();
    term_value.Vx().noalias() += cJx.transpose() * lamplus;
    term_value.Vxx() += cstr_data.Hxx_;
    term_value.Vxx().noalias() += cJx.transpose() * fb;
  }

  term_value.storage =
      term_value.storage.template selfadjointView<Eigen::Lower>();
}

template <typename Scalar>
bool SolverProxDDP<Scalar>::computeGains(const Problem &problem,
                                         Workspace &workspace, Results &results,
                                         const std::size_t t) const {
  const StageModel &stage = *problem.stages_[t];

  const VParams &vnext = workspace.value_params[t + 1];
  const QParams &qparam = workspace.q_params[t];

  StageData &stage_data = workspace.problem_data.getStageData(t);

  const int nprim = stage.numPrimal();
  const int ndual = stage.numDual();
  const int ndx1 = stage.ndx1();
  const int nu = stage.nu();
  const int ndx2 = stage.ndx2();

  assert(vnext.storage.rows() == ndx2 + 1);
  assert(vnext.storage.cols() == ndx2 + 1);

  const VectorXs &laminnr = results.lams[t + 1];
  const VectorXs &lamplus = workspace.lams_plus[t + 1];

  MatrixXs &kkt_mat = workspace.kkt_mat_buf_[t + 1];
  MatrixXs &kkt_rhs = workspace.kkt_rhs_buf_[t + 1];
  BlockXs kkt_jac = kkt_mat.block(nprim, 0, ndual, nprim);
  BlockXs kkt_top_left = kkt_mat.topLeftCorner(nprim, nprim);
  Eigen::Diagonal<BlockXs> kkt_low_right =
      kkt_mat.bottomRightCorner(ndual, ndual).diagonal();

  typename MatrixXs::ColXpr kkt_rhs_ff = kkt_rhs.col(0);
  auto kkt_rhs_fb = kkt_rhs.rightCols(ndx1);

  // blocks: u, y, and dual
  kkt_rhs_ff.head(nu) = qparam.Qu;
  kkt_rhs_ff.segment(nu, ndx2) = qparam.Qy;

  kkt_rhs_fb.topRows(nu) = qparam.Qxu.transpose();
  kkt_rhs_fb.middleRows(nu, ndx2) = qparam.Qxy.transpose();

  // KKT matrix: (u, y)-block = bottom right of q hessian
  kkt_top_left = qparam.hess_.bottomRightCorner(nprim, nprim);

  const ConstraintContainer<Scalar> &cstr_mgr = stage.constraints_;

  // Loop over constraints
  for (std::size_t j = 0; j < stage.numConstraints(); j++) {
    FunctionData &cstr_data = *stage_data.constraint_data[j];
    const auto laminnr_j = cstr_mgr.getConstSegmentByConstraint(laminnr, j);
    const auto lamplus_j = cstr_mgr.getConstSegmentByConstraint(lamplus, j);

    // update the KKT jacobian columns
    cstr_mgr.getBlockByConstraint(kkt_jac, j) =
        cstr_data.jac_buffer_.rightCols(nprim);
    cstr_mgr.getBlockByConstraint(kkt_rhs_fb.bottomRows(ndual), j) =
        cstr_data.Jx_;
    cstr_mgr.getSegmentByConstraint(kkt_rhs_ff.tail(ndual), j) =
        mu_scaled() * (lamplus_j - laminnr_j);

    kkt_low_right.array() = -mu_scaled();
  }

  /* Compute gains with LDLT */
  kkt_mat = kkt_mat.template selfadjointView<Eigen::Lower>();
  Eigen::LDLT<MatrixXs, Eigen::Lower> &ldlt = workspace.ldlts_[t + 1];
  ldlt.compute(kkt_mat);

  // check inertia
  {
    std::array<unsigned int, 3> inertia;
    math::compute_inertia(ldlt.vectorD(), inertia.data());
    const bool inertia_ok = (inertia[0] == (unsigned)nprim) &&
                            (inertia[1] == 0U) &&
                            (inertia[2] == (unsigned)ndual);
    if (!inertia_ok) {
#ifndef NDEBUG
      // print inertia
      fmt::print("[{:d}] kkt inertia ({})\n", t + 1, fmt::join(inertia, ","));
      fmt::print(" (reg={})\n", xreg_);
#endif
    }
    if (inertia[1] > 0U) {
      return false;
    }
    if (inertia[2] != (unsigned)ndual) {
      return false;
    }
  }

  MatrixXs &gains = results.gains_[t];
  gains = -kkt_rhs;
  ldlt.solveInPlace(gains);

  const Scalar resdl_thresh = 1e-10;
  const std::size_t MAX_REFINEMENT_STEPS = 5;
  MatrixXs &resdl = workspace.kkt_resdls_[t + 1];
  Scalar resdl_norm = 0.;
  for (std::size_t n = 0; n < MAX_REFINEMENT_STEPS; n++) {
    resdl = -(kkt_mat * gains + kkt_rhs);
    resdl_norm = math::infty_norm(resdl);
#ifndef NDEBUG
    std::FILE *fi = std::fopen("pddp.log", "a");
    fmt::print(fi, "[{:d}] Linear solve (try {:d}) resdl={:.3e}\n", t + 1, n,
               resdl_norm);
    std::fclose(fi);
#endif
    if (resdl_norm < resdl_thresh)
      break;
    ldlt.solveInPlace(resdl);
    gains += resdl;
  }

  /* Value function */
  VParams &vp = workspace.value_params[t];
  auto Qxw = kkt_rhs_fb.transpose();
  auto ff = results.getFeedforward(t);
  auto fb = results.getFeedback(t);

#ifndef NDEBUG
  std::FILE *fi = std::fopen("pddp.log", "a");
  if (t == workspace.nsteps - 1)
    fmt::print(fi, "[backward {:d}]\n", results.num_iters + 1);
  fmt::print(fi, "uff[{:d}]={}\n", t, ff.head(nu).transpose());
  fmt::print(fi, "V'x[{:d}]={}\n", t, vnext.Vx().transpose());
  auto lam_head = laminnr.head(vnext.Vx().size());
  fmt::print(fi, "l-V[{:d}]={}\n", t, (lam_head - vnext.Vx()).transpose());
  std::fclose(fi);
#endif
  vp.Vx() = qparam.Qx + Qxw * ff;
  vp.Vxx() = qparam.Qxx + Qxw * fb;
  vp.storage = vp.storage.template selfadjointView<Eigen::Lower>();
  return true;
}

template <typename Scalar>
bool SolverProxDDP<Scalar>::run(const Problem &problem,
                                const std::vector<VectorXs> &xs_init,
                                const std::vector<VectorXs> &us_init,
                                const std::vector<VectorXs> &lams_init) {
  if (workspace_ == 0 || results_ == 0) {
    proxddp_runtime_error("workspace and results were not allocated yet!");
  }
  Workspace &workspace = *workspace_;
  Results &results = *results_;

  checkTrajectoryAndAssign(problem, xs_init, us_init, results.xs, results.us);
  if (lams_init.size() == results.lams.size()) {
    for (std::size_t i = 0; i < lams_init.size(); i++) {
      long size = std::min(lams_init[i].rows(), results.lams[i].rows());
      results.lams[i].head(size) = lams_init[i].head(size);
    }
  }

#ifndef NDEBUG
  std::FILE *fi = std::fopen("pddp.log", "w");
  std::fclose(fi);
#endif

  logger.active = (verbose_ > 0);
  logger.start();

  setPenalty(mu_init);
  setRho(rho_init);
  xreg_ = reg_init;
  ureg_ = reg_init;

  workspace.prev_xs = results.xs;
  workspace.prev_us = results.us;
  workspace.prev_lams = results.lams;

  inner_tol_ = inner_tol0;
  prim_tol_ = prim_tol0;
  updateTolerancesOnFailure();

  inner_tol_ = std::max(inner_tol_, target_tol_);
  prim_tol_ = std::max(prim_tol_, target_tol_);

  bool &conv = results.conv;
  fmt::color colout = fmt::color::white;

  results.al_iter = 0;
  results.num_iters = 0;
  std::size_t &al_iter = results.al_iter;
  while ((al_iter < MAX_AL_ITERS) && (results.num_iters < MAX_ITERS)) {
    if (verbose_ >= 1) {
      fmt::print(fmt::emphasis::bold | fmt::fg(colout), "[AL iter {:>2d}]",
                 al_iter + 1);
      fmt::print(" ("
                 "inner_tol {:.2e} | "
                 "prim_tol  {:.2e} | "
                 "mu  {:.2g} | "
                 "rho {:.2g} )\n",
                 inner_tol_, prim_tol_, mu(), rho());
    }
    bool inner_conv = innerLoop(problem, workspace, results);
    if (!inner_conv) {
      fmt::print(fmt::fg(fmt::color::red), "Inner loop failed to converge.");
      fmt::print("\n");
      break;
    }

    // accept primal updates
    workspace.prev_xs = results.xs;
    workspace.prev_us = results.us;

    if (results.primal_infeasibility <= prim_tol_) {
      updateTolerancesOnSuccess();

      switch (multiplier_update_mode) {
      case MultiplierUpdateMode::NEWTON:
        workspace.prev_lams = results.lams;
        break;
      case MultiplierUpdateMode::PRIMAL:
        workspace.prev_lams = workspace.lams_plus;
        break;
      case MultiplierUpdateMode::PRIMAL_DUAL:
        workspace.prev_lams = workspace.lams_pdal;
        break;
      default:
        break;
      }

      if (std::max(results.primal_infeasibility, results.dual_infeasibility) <=
          target_tol_) {
        conv = true;
        break;
      }
      colout = fmt::color::dodger_blue;
    } else {
      bclUpdateALPenalty();
      updateTolerancesOnFailure();
      colout = fmt::color::red;
    }
    rho_penal_ *= bcl_params.rho_update_factor;

    inner_tol_ = std::max(inner_tol_, target_tol_);
    prim_tol_ = std::max(prim_tol_, target_tol_);

    al_iter++;
  }

  logger.finish(conv);
  invokeCallbacks(workspace, results);
  return conv;
}

template <typename Scalar>
bool SolverProxDDP<Scalar>::innerLoop(const Problem &problem,
                                      Workspace &workspace, Results &results) {

  // merit function evaluation
  auto merit_eval_lin = [&](Scalar a0) {
    tryStep(problem, workspace, results, a0);
    computeProxTerms(workspace.trial_xs, workspace.trial_us, workspace);
    computeMultipliers(problem, workspace, workspace.trial_lams,
                       workspace.trial_prob_data, false);
    return merit_fun.evaluate(problem, workspace.trial_lams, workspace,
                              workspace.trial_prob_data);
  };
  auto merit_eval_fun = [&](Scalar a0) {
    switch (this->rollout_type) {
    case RolloutType::LINEAR:
      return merit_eval_lin(a0);
      break;
    case RolloutType::NONLINEAR:
      nonlinearRollout(problem, workspace, results, a0);
      problem.evaluate(workspace.trial_xs, workspace.trial_us,
                       workspace.trial_prob_data);
      computeProxTerms(workspace.trial_xs, workspace.trial_us, workspace);
      computeMultipliers(problem, workspace, workspace.trial_lams,
                         workspace.trial_prob_data, false);
      return merit_fun.evaluate(problem, workspace.trial_lams, workspace,
                                workspace.trial_prob_data);
      break;
    default:
      proxddp_runtime_error("RolloutType unrecognized.");
      break;
    }
  };

  Scalar phi0 = 0.;
  const Scalar fd_eps = 1e-9;
  Scalar phieps = 0., dphi0 = 0.;

  logger.active = (verbose_ > 0);

  std::size_t &k = results.num_iters;
  while (k < MAX_ITERS) {
    problem.evaluate(results.xs, results.us, workspace.problem_data);
    problem.computeDerivatives(results.xs, results.us, workspace.problem_data);
    computeProxTerms(results.xs, results.us, workspace);
    computeProxDerivatives(results.xs, results.us, workspace);
    computeMultipliers(problem, workspace, results.lams, workspace.problem_data,
                       true);
    phi0 = merit_fun.evaluate(problem, results.lams, workspace,
                              workspace.problem_data);

    while (true) {
      bool success = backwardPass(problem, workspace, results);
      if (success) {
        break;
      } else {
        if (xreg_ == this->reg_max) {
          return false;
        }
        this->increase_reg();
        continue;
      }
    }

    computeInfeasibilities(problem, workspace, results);

    bool inner_conv = (workspace.inner_criterion < inner_tol_);
    if (inner_conv && (k >= 1)) {
      return true;
    }

    linearRollout(problem, workspace, results);
    phieps = merit_eval_lin(fd_eps);
    dphi0 = (phieps - phi0) / fd_eps;

    // otherwise continue linesearch
    Scalar alpha_opt = 1;

    Scalar phi_new = proxnlp::ArmijoLinesearch<Scalar>(ls_params).run(
        merit_eval_fun, phi0, dphi0, alpha_opt);
    results.traj_cost_ = merit_fun.traj_cost;
    results.merit_value_ = phi_new;

#ifndef NDEBUG
    {
      int nalph = 80;
      Scalar a = 0.;
      Scalar da = 1. / (nalph + 1);
      const auto fname = LS_DEBUG_LOG_PATH;
      std::FILE *file = 0;
      if (k == 0) {
        file = std::fopen(fname, "w");
        fmt::print(file, "k,alpha,phi\n");
      } else {
        file = std::fopen(fname, "a");
      }
      const char *fmtstr = "{:d}, {:.4e}, {:.5e}\n";
      for (int i = 0; i <= nalph + 1; i++) {
        Scalar p = merit_eval_fun(a);
        fmt::print(file, fmtstr, k, a, p);
        a += da;
      }
      if (alpha_opt < da) {
        nalph = 40.;
        VectorXs als;
        als.setLinSpaced(nalph, 0., 2 * alpha_opt);
        for (int i = 1; i < als.size(); i++) {
          fmt::print(file, fmtstr, k, als(i), merit_eval_fun(als(i)));
        }
      }
      fmt::print(file, fmtstr, k, alpha_opt, merit_eval_fun(alpha_opt));
      std::fclose(file);
    }
#endif
    // accept the step
    results.xs = workspace.trial_xs;
    results.us = workspace.trial_us;
    results.lams = workspace.trial_lams;
    PROXDDP_RAISE_IF_NAN_NAME(alpha_opt, "alpha_opt");
    PROXDDP_RAISE_IF_NAN_NAME(results.merit_value_, "results.merit_value");
    PROXDDP_RAISE_IF_NAN_NAME(results.traj_cost_, "results.traj_cost");

    LogRecord iter_log;
    iter_log.iter = k + 1;
    iter_log.xreg = xreg_;
    iter_log.inner_crit = workspace.inner_criterion;
    iter_log.prim_err = results.primal_infeasibility;
    iter_log.dual_err = results.dual_infeasibility;
    iter_log.step_size = alpha_opt;
    iter_log.dphi0 = dphi0;
    iter_log.merit = results.merit_value_;
    iter_log.dM = results.merit_value_ - phi0;

    logger.log(iter_log);

    if (std::abs(dphi0) <= ls_params.dphi_thresh)
      return true;
    if (alpha_opt == ls_params.alpha_min)
      this->increase_reg();

    invokeCallbacks(workspace, results);

    k++;
  }
  return false;
}

template <typename Scalar>
void SolverProxDDP<Scalar>::computeInfeasibilities(const Problem &problem,
                                                   Workspace &workspace,
                                                   Results &results) const {
  const TrajOptDataTpl<Scalar> &prob_data = workspace.problem_data;
  const std::size_t nsteps = problem.numSteps();
  {
    const FunctionData &init_data = prob_data.getInitData();
    workspace.primal_infeas_by_stage(0) = math::infty_norm(init_data.value_);
  }

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];
    const StageData &stage_data = prob_data.getStageData(i);
    Scalar infeas_over_j = 0.;
    for (std::size_t j = 0; j < stage.numConstraints(); j++) {
      const ConstraintSetBase<Scalar> &cstr_set =
          stage.constraints_.getConstraintSet(j);
      auto &v = stage_data.constraint_data[j]->value_;
      cstr_set.normalConeProjection(v, v);
      infeas_over_j = std::max(infeas_over_j, math::infty_norm(v));
    }
    workspace.primal_infeas_by_stage(long(i + 1)) = infeas_over_j;
  }
  if (problem.term_constraint_) {
    const FunctionData &data = *prob_data.term_cstr_data;
    workspace.primal_infeas_by_stage((long)nsteps + 1) =
        math::infty_norm(data.value_);
  }

  results.primal_infeasibility =
      math::infty_norm(workspace.primal_infeas_by_stage);

  for (std::size_t i = 1; i <= nsteps; i++) {
    const StageModel &st = *problem.stages_[i - 1];
    const int nu = st.nu();
    const int ndx2 = st.ndx2();
    const int ndual = st.numDual();
    Scalar ru, ry, rl;
    const auto kkt_rhs_0 = workspace.kkt_rhs_buf_[i].col(0);
    const auto kktlam = kkt_rhs_0.tail(ndual);

    const VParams &vp = workspace.value_params[i];
    const QParams &qpar = workspace.q_params[i - 1];

    decltype(auto) gu = qpar.Qu;
    decltype(auto) gy = qpar.Qy;
    ru = math::infty_norm(gu);
    ry = math::infty_norm(gy);
    rl = math::infty_norm(kktlam);
    workspace.inner_criterion_by_stage(long(i)) = std::max({ru, ry, rl});
    workspace.inner_criterion_x = std::max(workspace.inner_criterion_x, ry);
    workspace.inner_criterion_u = std::max(workspace.inner_criterion_u, ru);
    workspace.inner_criterion_l = std::max(workspace.inner_criterion_l, rl);
#ifndef NDEBUG
    Scalar ru_bis_wtf_cmon;
    {
      const StageData &sd = prob_data.getStageData(i - 1);
      auto gu_bis = sd.cost_data->Lu_;

      const DynamicsDataTpl<Scalar> &dd = sd.dyn_data();
      gu_bis += dd.Ju_.transpose() * vp.Vx();
      ru_bis_wtf_cmon = math::infty_norm(gu_bis);
    }
    std::FILE *fi = std::fopen("pddp.log", "a");
    fmt::print(fi, "[{:>3d}]ru={:.2e},ry={:.2e},rl={:.2e},", i, ru, ry, rl);
    fmt::print(fi, " ru_other={:.3e}\n", ru_bis_wtf_cmon);
    std::fclose(fi);
#endif
    {
      const CostData &proxdata = *workspace.prox_datas[i - 1];
      const CostData &proxnext = *workspace.prox_datas[i];
      auto gu_non_reg = gu - rho() * proxdata.Lu_;
      auto gy_non_reg = gy - rho() * proxnext.Lx_;
      Scalar dual_res_u = math::infty_norm(gu_non_reg);
      Scalar dual_res_y = math::infty_norm(gy_non_reg);
      workspace.dual_infeas_by_stage(long(i)) =
          std::max(dual_res_u, dual_res_y);
    }
  }
  workspace.inner_criterion =
      math::infty_norm(workspace.inner_criterion_by_stage);
  results.dual_infeasibility = math::infty_norm(workspace.dual_infeas_by_stage);

  return;
}

} // namespace proxddp
