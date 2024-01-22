/// @file solver-proxddp.hxx
/// @brief  Implementations for the trajectory optimization algorithm.
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "solver-proxddp.hpp"
#include "aligator/core/lagrangian.hpp"
#ifndef NDEBUG
#include <fmt/ostream.h>
#endif

namespace aligator {

template <typename Scalar>
void computeProjectedJacobians(const TrajOptProblemTpl<Scalar> &problem,
                               WorkspaceTpl<Scalar> &workspace) {
  using ProductOp = ConstraintSetProductTpl<Scalar>;
  auto &sif = workspace.shifted_constraints;

  const TrajOptDataTpl<Scalar> &prob_data = workspace.problem_data;
  const std::size_t nsteps = workspace.nsteps;
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModelTpl<Scalar> &sm = *problem.stages_[i];
    const StageDataTpl<Scalar> &sd = *prob_data.stage_data[i];
    auto &jac = workspace.constraintProjJacobians[i];

    for (std::size_t j = 0; j < sm.numConstraints(); j++) {
      jac(j, 0) = sd.constraint_data[j]->Jx_;
      jac(j, 1) = sd.constraint_data[j]->Ju_;
    }

    const ProductOp &op = workspace.constraintProductOperators[i];
    op.applyNormalConeProjectionJacobian(sif[i], jac.matrix());
  }

  if (!problem.term_cstrs_.empty()) {
    auto &jac = workspace.constraintProjJacobians[nsteps];
    const auto &cds = prob_data.term_cstr_data;
    for (std::size_t j = 0; j < cds.size(); j++) {
      jac(j, 0) = cds[j]->Jx_;
    }

    const ProductOp &op = workspace.constraintProductOperators[nsteps];
    op.applyNormalConeProjectionJacobian(sif[nsteps], jac.matrix());
  }
}

template <typename Scalar>
SolverProxDDPTpl<Scalar>::SolverProxDDPTpl(const Scalar tol,
                                           const Scalar mu_init,
                                           const Scalar rho_init,
                                           const std::size_t max_iters,
                                           VerboseLevel verbose,
                                           HessianApprox hess_approx)
    : target_tol_(tol), mu_init(mu_init), rho_init(rho_init), verbose_(verbose),
      hess_approx_(hess_approx), max_iters(max_iters), rollout_max_iters(1),
      linesearch_(ls_params) {
  ls_params.interp_type = proxsuite::nlp::LSInterpolation::CUBIC;
}

template <typename Scalar>
Scalar SolverProxDDPTpl<Scalar>::forward_linear_impl(const Problem &problem,
                                                     Workspace &workspace,
                                                     const Results &results,
                                                     const Scalar alpha) {

  const std::size_t nsteps = workspace.nsteps;
  assert(results.xs.size() == nsteps + 1);
  assert(results.us.size() == nsteps);
  assert(results.lams.size() == nsteps + 1);
  assert(results.vs.size() == nsteps + 1);

  vectorMultiplyAdd(results.lams, workspace.dlams, workspace.trial_lams, alpha);
  vectorMultiplyAdd(results.vs, workspace.dvs, workspace.trial_vs, alpha);

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];
    stage.xspace_->integrate(results.xs[i], alpha * workspace.dxs[i],
                             workspace.trial_xs[i]);
    stage.uspace_->integrate(results.us[i], alpha * workspace.dus[i],
                             workspace.trial_us[i]);
  }
  const StageModel &stage = *problem.stages_[nsteps - 1];
  stage.xspace_next_->integrate(results.xs[nsteps],
                                alpha * workspace.dxs[nsteps],
                                workspace.trial_xs[nsteps]);
  TrajOptData &prob_data = workspace.problem_data;
  problem.evaluate(workspace.trial_xs, workspace.trial_us, prob_data);
  prob_data.cost_ = problem.computeTrajectoryCost(prob_data);
  return prob_data.cost_;
}

template <typename Scalar>
void SolverProxDDPTpl<Scalar>::setup(const Problem &problem) {
  workspace_ = Workspace(problem);
  results_ = Results(problem);
  linesearch_.setOptions(ls_params);

  workspace_.configureScalers(problem, mu_penal_,
                              applyDefaultScalingStrategy<Scalar>);
  linearSolver_ = std::make_unique<gar::ProximalRiccatiSolver<Scalar>>(
      workspace_.lqr_problem);
}

/// TODO: REWORK FOR NEW MULTIPLIERS
template <typename Scalar>
void SolverProxDDPTpl<Scalar>::computeMultipliers(
    const Problem &problem, const std::vector<VectorXs> &lams,
    const std::vector<VectorXs> &vs) {
  using BlkView = BlkMatrix<VectorRef, -1, 1>;

  const TrajOptData &prob_data = workspace_.problem_data;
  const std::size_t nsteps = workspace_.nsteps;

  const std::vector<VectorXs> &lams_prev = workspace_.prev_lams;
  std::vector<VectorXs> &lams_plus = workspace_.lams_plus;
  std::vector<VectorXs> &lams_pdal = workspace_.lams_pdal;

  const std::vector<VectorXs> &vs_prev = workspace_.prev_vs;
  std::vector<VectorXs> &vs_plus = workspace_.vs_plus;
  std::vector<VectorXs> &vs_pdal = workspace_.vs_pdal;

  std::vector<VectorXs> &Lds = workspace_.Lds_;
  std::vector<VectorXs> &Lvs = workspace_.Lvs_;
  std::vector<VectorXs> &shifted_constraints = workspace_.shifted_constraints;

  assert(Lds.size() == lams_prev.size());
  assert(Lds.size() == nsteps + 1);
  assert(Lvs.size() == vs_prev.size());
  assert(Lvs.size() == nsteps + 1);

  // initial constraint
  {
    StageFunctionData &dd = *prob_data.init_data;
    lams_plus[0] = lams_prev[0] + mu_inv() * dd.value_;
    lams_pdal[0] = 2 * lams_plus[0] - lams[0];
    /// TODO: generalize to the other types of initial constraint (non-equality)
    workspace_.dyn_slacks[0] = dd.value_;
    Lds[0] = mu() * (lams_plus[0] - lams[0]);
    ALIGATOR_RAISE_IF_NAN(Lds[0]);
  }

  // loop over the stages
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];
    const StageData &sd = *prob_data.stage_data[i];
    const StageFunctionData &dd = *sd.dynamics_data;
    const ConstraintStack &cstr_stack = stage.constraints_;
    const CstrProximalScaler &scaler = workspace_.cstr_scalers[i];

    assert(vs[i].size() == stage.nc());
    assert(lams[i + 1].size() == stage.ndx2());

    // 1. compute shifted dynamics error
    workspace_.dyn_slacks[i + 1] = dd.value_;
    lams_plus[i + 1] = lams_prev[i + 1] + mu_inv() * dd.value_;
    lams_pdal[i + 1] = 2 * lams_plus[i + 1] - lams[i + 1];
    Lds[i + 1] = mu() * (lams_plus[i + 1] - lams[i + 1]);
    ALIGATOR_RAISE_IF_NAN(Lds[i + 1]);

    // 2. use product constraint operator
    // to compute the new multiplier estimates
    const ConstraintSetProductTpl<Scalar> &op =
        workspace_.constraintProductOperators[i];

    // fill in shifted constraints buffer
    BlkView scvView(shifted_constraints[i], cstr_stack.dims());
    for (size_t j = 0; j < cstr_stack.size(); j++) {
      const StageFunctionData &cd = *sd.constraint_data[j];
      scvView[j] = cd.value_;
    }
    vs_plus[i] = vs_prev[i];
    vs_plus[i].noalias() += scaler.applyInverse(shifted_constraints[i]);
    shifted_constraints[i] += scaler.apply(vs_prev[i]); // apply shift
    op.normalConeProjection(shifted_constraints[i], vs_plus[i]);
    op.computeActiveSet(shifted_constraints[i],
                        workspace_.active_constraints[i]);
    Lvs[i].noalias() = scaler.apply(vs_plus[i] - vs[i]);
    assert(Lvs[i].size() == stage.nc());
    ALIGATOR_RAISE_IF_NAN(Lvs[i]);
  }

  if (!problem.term_cstrs_.empty()) {
    assert(problem.term_cstrs_.size() == prob_data.term_cstr_data.size());
    const ConstraintStack &cstr_stack = problem.term_cstrs_;
    const CstrProximalScaler &scaler = workspace_.cstr_scalers[nsteps];

    const ConstraintSetProductTpl<Scalar> &op =
        workspace_.constraintProductOperators[nsteps];

    BlkView scvView(shifted_constraints[nsteps], cstr_stack.dims());
    for (size_t j = 0; j < cstr_stack.size(); j++) {
      const StageFunctionData &cd = *prob_data.term_cstr_data[j];
      scvView[j] = cd.value_;
    }
    vs_plus[nsteps] =
        vs_prev[nsteps] + scaler.applyInverse(shifted_constraints[nsteps]);
    op.normalConeProjection(vs_plus[nsteps], vs_plus[nsteps]);
    shifted_constraints[nsteps] += scaler.apply(vs_prev[nsteps]);
    op.computeActiveSet(shifted_constraints[nsteps],
                        workspace_.active_constraints[nsteps]);
    Lvs[nsteps].noalias() = scaler.apply(vs_plus[nsteps] - vs[nsteps]);
    assert(Lvs[nsteps].size() == cstr_stack.totalDim());
    ALIGATOR_RAISE_IF_NAN(Lvs[nsteps]);
  }
}

template <typename Scalar>
Scalar SolverProxDDPTpl<Scalar>::nonlinear_rollout_impl(const Problem &problem,
                                                        const Scalar alpha) {
  using ExplicitDynData = ExplicitDynamicsDataTpl<Scalar>;

  const std::size_t nsteps = workspace_.nsteps;
  std::vector<VectorXs> &xs = workspace_.trial_xs;
  std::vector<VectorXs> &us = workspace_.trial_us;
  std::vector<VectorXs> &lams = workspace_.trial_lams;
  std::vector<VectorXs> &dxs = workspace_.dxs;
  std::vector<VectorXs> &dus = workspace_.dus;
  const std::vector<VectorXs> &lams_prev = workspace_.prev_lams;
  std::vector<VectorXs> &dyn_slacks = workspace_.dyn_slacks;
  TrajOptData &prob_data = workspace_.problem_data;

  {
    const StageModel &stage = *problem.stages_[0];
    // use lams[0] as a tmp var for alpha * dx0
    lams[0] = alpha * dxs[0];
    stage.xspace().integrate(results_.xs[0], lams[0], xs[0]);
    lams[0] = results_.lams[0] + alpha * workspace_.dlams[0];

    ALIGATOR_RAISE_IF_NAN_NAME(xs[0], fmt::format("xs[{:d}]", 0));
  }

  for (std::size_t t = 0; t < nsteps; t++) {
    const StageModel &stage = *problem.stages_[t];
    StageData &data = *prob_data.stage_data[t];

    const int nu = stage.nu();
    const int ndual = stage.numDual();

    ConstVectorRef ff = results_.getFeedforward(t);
    ConstMatrixRef fb = results_.getFeedback(t);
    auto ff_u = ff.head(nu);
    auto fb_u = fb.topRows(nu);
    auto ff_lm = ff.tail(ndual);
    auto fb_lm = fb.bottomRows(ndual);

    dus[t] = alpha * ff_u;
    dus[t].noalias() += fb_u * dxs[t];
    stage.uspace().integrate(results_.us[t], dus[t], us[t]);

    VectorRef dlam = workspace_.dlams[t + 1];
    dlam = alpha * ff_lm;
    dlam.noalias() += fb_lm * dxs[t];
    lams[t + 1] = results_.lams[t + 1] + dlam;

    stage.evaluate(xs[t], us[t], xs[t + 1], data);

    // compute desired multiple-shooting gap from the multipliers
    dyn_slacks[t] = mu() * (lams[t + 1] - lams_prev[t + 1]);

    DynamicsData &dd = *data.dynamics_data;

    // lambda to be called in both branches
    auto explicit_model_update_xnext = [&]() {
      ExplicitDynData &exp_dd = static_cast<ExplicitDynData &>(dd);
      stage.xspace_next().integrate(exp_dd.xnext_, dyn_slacks[t], xs[t + 1]);
      // at xs[i+1], the dynamics gap = the slack dyn_slack[i].
      exp_dd.value_ = -dyn_slacks[t];
    };

    if (!stage.has_dyn_model() || stage.dyn_model().is_explicit()) {
      explicit_model_update_xnext();
    } else {
      forwardDynamics<Scalar>::run(stage.dyn_model(), xs[t], us[t], dd,
                                   xs[t + 1], dyn_slacks[t], rollout_max_iters);
    }

    stage.xspace_next().difference(results_.xs[t + 1], xs[t + 1], dxs[t + 1]);

    ALIGATOR_RAISE_IF_NAN_NAME(xs[t + 1], fmt::format("xs[{:d}]", t + 1));
    ALIGATOR_RAISE_IF_NAN_NAME(us[t], fmt::format("us[{:d}]", t));
    ALIGATOR_RAISE_IF_NAN_NAME(lams[t + 1], fmt::format("lams[{:d}]", t + 1));
  }

  // TERMINAL NODE
  problem.term_cost_->evaluate(xs[nsteps], problem.unone_,
                               *prob_data.term_cost_data);

  for (std::size_t k = 0; k < problem.term_cstrs_.size(); ++k) {
    const ConstraintType &tc = problem.term_cstrs_[k];
    StageFunctionData &td = *prob_data.term_cstr_data[k];
    tc.func->evaluate(xs[nsteps], problem.unone_, xs[nsteps], td);
  }

  // update multiplier
  if (!problem.term_cstrs_.empty()) {
    VectorRef dlam = workspace_.dlams.back();
    ConstVectorRef dx = workspace_.dxs.back();
    auto ff = results_.getFeedforward(nsteps);
    auto fb = results_.getFeedback(nsteps);
    dlam = alpha * ff;
    dlam.noalias() += fb * dx;
    lams.back() = results_.lams.back() + dlam;
  }

  prob_data.cost_ = problem.computeTrajectoryCost(prob_data);
  return prob_data.cost_;
}

template <typename Scalar>
bool SolverProxDDPTpl<Scalar>::run(const Problem &problem,
                                   const std::vector<VectorXs> &xs_init,
                                   const std::vector<VectorXs> &us_init,
                                   const std::vector<VectorXs> &lams_init) {
  if (!workspace_.isInitialized() || !results_.isInitialized()) {
    ALIGATOR_RUNTIME_ERROR("workspace and results were not allocated yet!");
  }

  check_trajectory_and_assign(problem, xs_init, us_init, results_.xs,
                              results_.us);
  if (lams_init.size() == results_.lams.size()) {
    for (std::size_t i = 0; i < lams_init.size(); i++) {
      long size = std::min(lams_init[i].rows(), results_.lams[i].rows());
      results_.lams[i].head(size) = lams_init[i].head(size);
    }
  }

  if (force_initial_condition_) {
    workspace_.trial_xs[0] = problem.getInitState();
    workspace_.trial_lams[0].setZero();
  }

  logger.active = (verbose_ > 0);
  logger.printHeadline();

  setAlmPenalty(mu_init);
  setRho(rho_init);

  workspace_.prev_xs = results_.xs;
  workspace_.prev_us = results_.us;
  workspace_.prev_vs = results_.vs;
  workspace_.prev_lams = results_.lams;

  inner_tol_ = inner_tol0;
  prim_tol_ = prim_tol0;
  updateTolsOnFailure();

  inner_tol_ = std::max(inner_tol_, target_tol_);
  prim_tol_ = std::max(prim_tol_, target_tol_);

  bool &conv = results_.conv = false;

  results_.al_iter = 0;
  results_.num_iters = 0;
  std::size_t &al_iter = results_.al_iter;
  while ((al_iter < max_al_iters) && (results_.num_iters < max_iters)) {
    bool inner_conv = innerLoop(problem);
    if (!inner_conv) {
      al_iter++;
      break;
    }

    // accept primal updates
    workspace_.prev_xs = results_.xs;
    workspace_.prev_us = results_.us;

    if (results_.prim_infeas <= prim_tol_) {
      updateTolsOnSuccess();

      switch (multiplier_update_mode) {
      case MultiplierUpdateMode::NEWTON:
        workspace_.prev_vs = results_.vs;
        workspace_.prev_lams = results_.lams;
        break;
      case MultiplierUpdateMode::PRIMAL:
        workspace_.prev_vs = workspace_.vs_plus;
        workspace_.prev_lams = workspace_.lams_plus;
        break;
      case MultiplierUpdateMode::PRIMAL_DUAL:
        workspace_.prev_vs = workspace_.vs_pdal;
        workspace_.prev_lams = workspace_.lams_pdal;
        break;
      default:
        break;
      }

      Scalar criterion = std::max(results_.dual_infeas, results_.prim_infeas);
      if (criterion <= target_tol_) {
        conv = true;
        break;
      }
    } else {
      Scalar old_mu = mu_penal_;
      bclUpdateAlmPenalty();
      updateTolsOnFailure();
      if (math::scalar_close(old_mu, mu_penal_)) {
        // reset penalty to initial value
        setAlmPenalty(mu_init);
      }
    }
    rho_penal_ *= bcl_params.rho_update_factor;

    inner_tol_ = std::max(inner_tol_, 0.01 * target_tol_);
    prim_tol_ = std::max(prim_tol_, target_tol_);

    al_iter++;
  }

  logger.finish(conv);
  return conv;
}

template <typename Scalar>
Scalar SolverProxDDPTpl<Scalar>::forwardPass(const Problem &problem,
                                             const Scalar alpha) {
  switch (rollout_type_) {
  case RolloutType::LINEAR:
    forward_linear_impl(problem, workspace_, results_, alpha);
    break;
  case RolloutType::NONLINEAR:
    nonlinear_rollout_impl(problem, alpha);
    break;
  default:
    assert(false && "unknown RolloutType!");
    break;
  }
  computeMultipliers(problem, workspace_.trial_lams, workspace_.trial_vs);
  return PDALFunction<Scalar>::evaluate(mu(), problem, workspace_.trial_lams,
                                        workspace_.trial_vs, workspace_);
}

template <typename Scalar>
bool SolverProxDDPTpl<Scalar>::innerLoop(const Problem &problem) {

  auto merit_eval_fun = [&](Scalar a0) -> Scalar {
    return forwardPass(problem, a0);
  };

  LogRecord iter_log;

  std::size_t &iter = results_.num_iters;
  results_.traj_cost_ =
      problem.evaluate(results_.xs, results_.us, workspace_.problem_data);
  computeMultipliers(problem, results_.lams, results_.vs);
  results_.merit_value_ = PDALFunction<Scalar>::evaluate(
      mu(), problem, results_.lams, results_.vs, workspace_);

  for (; iter < max_iters; iter++) {
    // ASSUMPTION: last evaluation in previous iterate
    // was during linesearch, at the current candidate solution (x,u).
    /// TODO: make this smarter using e.g. some caching mechanism
    problem.computeDerivatives(results_.xs, results_.us,
                               workspace_.problem_data);
    computeProjectedJacobians(problem, workspace_);
    const Scalar phi0 = results_.merit_value_;

    LagrangianDerivatives<Scalar>::compute(problem, workspace_.problem_data,
                                           results_.lams, results_.vs,
                                           workspace_.Lxs_, workspace_.Lus_);
    if (force_initial_condition_) {
      workspace_.Lxs_[0].setZero();
      workspace_.Lds_[0].setZero();
    }
    computeInfeasibilities(problem);
    computeCriterion();

    Scalar outer_crit = std::max(results_.dual_infeas, results_.prim_infeas);
    if (outer_crit <= target_tol_)
      return true;

    initializeRegularization();
    updateLQSubproblem();
    linearSolver_->backward(mu_penal_, mu_penal_);

    bool inner_conv = (workspace_.inner_criterion <= inner_tol_);
    if (inner_conv)
      return true;

    linearSolver_->forward(workspace_.dxs, workspace_.dus, workspace_.dvs,
                           workspace_.dlams);
    if (force_initial_condition_) {
      workspace_.dxs[0].setZero();
      workspace_.dlams[0].setZero();
    }
    Scalar dphi0 = PDALFunction<Scalar>::directionalDerivative(
        mu(), problem, results_.lams, results_.vs, workspace_);
    ALIGATOR_RAISE_IF_NAN(dphi0);

    // check if we can early stop
    if (std::abs(dphi0) <= ls_params.dphi_thresh)
      return true;

    // otherwise continue linesearch
    Scalar alpha_opt = 1;
    Scalar phi_new = linesearch_.run(merit_eval_fun, phi0, dphi0, alpha_opt);

    // accept the step
    results_.xs = workspace_.trial_xs;
    results_.us = workspace_.trial_us;
    results_.vs = workspace_.trial_vs;
    results_.lams = workspace_.trial_lams;
    results_.traj_cost_ = workspace_.problem_data.cost_;
    results_.merit_value_ = phi_new;
    ALIGATOR_RAISE_IF_NAN_NAME(alpha_opt, "alpha_opt");
    ALIGATOR_RAISE_IF_NAN_NAME(results_.merit_value_, "results.merit_value");
    ALIGATOR_RAISE_IF_NAN_NAME(results_.traj_cost_, "results.traj_cost");

    iter_log.iter = iter + 1;
    iter_log.al_iter = results_.al_iter + 1;
    iter_log.xreg = xreg_;
    iter_log.inner_crit = workspace_.inner_criterion;
    iter_log.prim_err = results_.prim_infeas;
    iter_log.dual_err = results_.dual_infeas;
    iter_log.step_size = alpha_opt;
    iter_log.dphi0 = dphi0;
    iter_log.merit = phi_new;
    iter_log.dM = phi_new - phi0;
    iter_log.mu = mu();

    if (alpha_opt <= ls_params.alpha_min) {
      if (xreg_ >= reg_max)
        return false;
      increaseRegularization();
    }
    invokeCallbacks(workspace_, results_);
    logger.log(iter_log);

    xreg_last_ = xreg_;
  }
  return false;
}

template <typename Scalar>
void SolverProxDDPTpl<Scalar>::computeInfeasibilities(const Problem &problem) {
  // modifying quantities such as Qu, Qy... is allowed
  ALIGATOR_NOMALLOC_BEGIN;
  const std::size_t nsteps = workspace_.nsteps;

  std::vector<VectorXs> &lams_plus = workspace_.lams_plus;
  std::vector<VectorXs> &lams_prev = workspace_.prev_lams;
  std::vector<VectorXs> &vs_plus = workspace_.vs_plus;
  std::vector<VectorXs> &vs_prev = workspace_.prev_vs;
  std::vector<VectorXs> &stage_infeas = workspace_.stage_infeasibilities;

  // compute infeasibility of all stage constraints
  for (std::size_t i = 0; i < nsteps; i++) {
    const CstrProximalScaler &scaler = workspace_.cstr_scalers[i];
    stage_infeas[i] = vs_plus[i] - vs_prev[i];
    stage_infeas[i] = scaler.apply(stage_infeas[i]);

    workspace_.stage_cstr_violations[long(i)] =
        math::infty_norm(stage_infeas[i]);
  }

  // compute infeasibility of terminal constraints
  if (!problem.term_cstrs_.empty()) {
    const CstrProximalScaler &scaler = workspace_.cstr_scalers[nsteps];
    stage_infeas[nsteps] = vs_plus[nsteps] - vs_prev[nsteps];
    stage_infeas[nsteps] = scaler.apply(stage_infeas[nsteps]);

    workspace_.stage_cstr_violations[long(nsteps)] =
        math::infty_norm(stage_infeas[nsteps]);
  }

  results_.prim_infeas = std::max(math::infty_norm(stage_infeas),
                                  math::infty_norm(workspace_.dyn_slacks));

  ALIGATOR_NOMALLOC_END;
}

template <typename Scalar> void SolverProxDDPTpl<Scalar>::computeCriterion() {
  const std::size_t nsteps = workspace_.nsteps;

  workspace_.stage_inner_crits.setZero();

  for (std::size_t i = 0; i < nsteps; i++) {
    Scalar rx = math::infty_norm(workspace_.Lxs_[i]);
    Scalar ru = math::infty_norm(workspace_.Lus_[i]);
    Scalar rd = math::infty_norm(workspace_.Lds_[i]);
    Scalar rc = math::infty_norm(workspace_.Lvs_[i]);

    workspace_.stage_inner_crits[long(i)] = std::max({rx, ru, rd, rc});
    workspace_.state_dual_infeas[long(i)] = rx;
    workspace_.control_dual_infeas[long(i)] = ru;
  }
  workspace_.state_dual_infeas[long(nsteps)] =
      math::infty_norm(workspace_.Lxs_[nsteps]);

  workspace_.inner_criterion = math::infty_norm(workspace_.stage_inner_crits);
  results_.dual_infeas =
      std::max(math::infty_norm(workspace_.state_dual_infeas),
               math::infty_norm(workspace_.control_dual_infeas));
}

template <typename Scalar> void SolverProxDDPTpl<Scalar>::updateLQSubproblem() {
  LQProblem &prob = workspace_.lqr_problem;
  const TrajOptData &pd = workspace_.problem_data;

  using gar::LQRKnotTpl;

  size_t N = (size_t)prob.horizon();
  assert(N == workspace_.nsteps);

  for (size_t t = 0; t < N; t++) {
    const StageData &sd = *pd.stage_data[t];
    LQRKnotTpl<Scalar> &knot = prob.stages[t];
    const StageFunctionData &dd = *sd.dynamics_data;
    const CostData &cd = *sd.cost_data;
    const CstrProximalScaler &sc = workspace_.cstr_scalers[t];

    knot.A = dd.Jx_;
    knot.B = dd.Ju_;
    knot.E = dd.Jy_;
    knot.f = workspace_.Lds_[t + 1];

    knot.Q = cd.Lxx_;
    knot.S = cd.Lxu_;
    knot.R = cd.Luu_;
    knot.q = workspace_.Lxs_[t];
    knot.r = workspace_.Lus_[t];

    // dynamics hessians
    if (hess_approx_ == HessianApprox::EXACT) {
      knot.Q += dd.Hxx_;
      knot.S += dd.Hxu_;
      knot.R += dd.Huu_;
    }

    // TODO: handle the bloody constraints
    assert(knot.nc == workspace_.constraintProjJacobians[t].rows());
    knot.C = workspace_.constraintProjJacobians[t].blockCol(0);
    knot.D = workspace_.constraintProjJacobians[t].blockCol(1);
    knot.d = workspace_.Lvs_[t];

    // correct right-hand side
    knot.q.noalias() += knot.C.transpose() * sc.apply(knot.d);
    knot.r.noalias() += knot.D.transpose() * sc.apply(knot.d);
  }

  {
    LQRKnotTpl<Scalar> &knot = prob.stages[N];
    const CostData &tcd = *pd.term_cost_data;
    knot.Q = tcd.Lxx_;
    knot.q = workspace_.Lxs_[N];
    knot.C = workspace_.constraintProjJacobians[N].blockCol(0);
    knot.d = workspace_.Lvs_[N];
  }

  const StageFunctionData &id = *pd.init_data;
  prob.G0 = id.Jx_;
  prob.g0.noalias() = workspace_.Lds_[0];

  LQRKnotTpl<Scalar> &model = prob.stages[0];
  model.Q += id.Hxx_;
}

} // namespace aligator
