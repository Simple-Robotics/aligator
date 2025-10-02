/// @file solver-proxddp.hxx
/// @brief  Implementations for the trajectory optimization algorithm.
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "solver-proxddp.hpp"
#include "merit-function.hpp"
#include "aligator/core/lagrangian.hpp"
#include "aligator/threads.hpp"
#include "aligator/core/explicit-dynamics.hpp"

#include "aligator/gar/proximal-riccati.hpp"
#include "aligator/gar/parallel-solver.hpp"
#include "aligator/gar/dense-riccati.hpp"

#include "aligator/tracy.hpp"

namespace aligator {

// [1], realted to Appendix A, details on aug. Lagrangian method
// interpretation as shifted-penalty method
template <typename Scalar>
void computeProjectedJacobians(const TrajOptProblemTpl<Scalar> &problem,
                               const Scalar mu_inv,
                               WorkspaceTpl<Scalar> &workspace) {
  ALIGATOR_TRACY_ZONE_SCOPED;
  using ProductOp = ConstraintSetProductTpl<Scalar>;
  auto &sif = workspace.shifted_constraints;

  const TrajOptDataTpl<Scalar> &prob_data = workspace.problem_data;
  const size_t N = workspace.nsteps;
  for (size_t i = 0; i < N; i++) {
    const StageModelTpl<Scalar> &sm = *problem.stages_[i];
    const StageDataTpl<Scalar> &sd = *prob_data.stage_data[i];
    auto &jac = workspace.cstr_proj_jacs[i];

    for (size_t j = 0; j < sm.numConstraints(); j++) {
      jac(j, 0) = sd.constraint_data[j]->Jx_;
      jac(j, 1) = sd.constraint_data[j]->Ju_;
    }

    auto Px = jac.blockCol(0);
    auto Pu = jac.blockCol(1);
    auto Lv = workspace.Lvs[i] * mu_inv;
    workspace.cstr_lx_corr[i].noalias() = Px.transpose() * Lv;
    workspace.cstr_lu_corr[i].noalias() = Pu.transpose() * Lv;
    const ProductOp &op = workspace.cstr_product_sets[i];
    op.applyNormalConeProjectionJacobian(sif[i], jac.matrix());
    workspace.cstr_lx_corr[i].noalias() -= Px.transpose() * Lv;
    workspace.cstr_lu_corr[i].noalias() -= Pu.transpose() * Lv;
  }

  if (!problem.term_cstrs_.empty()) {
    auto &jac = workspace.cstr_proj_jacs[N];
    const auto &cds = prob_data.term_cstr_data;
    for (size_t j = 0; j < cds.size(); j++) {
      jac(j, 0) = cds[j]->Jx_;
    }

    auto Px = jac.blockCol(0);
    auto Lv = workspace.Lvs[N] * mu_inv;
    workspace.cstr_lx_corr[N].noalias() = Px.transpose() * Lv;
    const ProductOp &op = workspace.cstr_product_sets[N];
    op.applyNormalConeProjectionJacobian(sif[N], jac.matrix());
    workspace.cstr_lx_corr[N].noalias() -= Px.transpose() * Lv;
  }
}

template <typename Scalar>
SolverProxDDPTpl<Scalar>::SolverProxDDPTpl(const Scalar tol,
                                           const Scalar mu_init,
                                           const size_t max_iters,
                                           VerboseLevel verbose,
                                           StepAcceptanceStrategy sa_strategy,
                                           HessianApprox hess_approx)
    : target_tol_(tol)
    , target_dual_tol_(tol)
    , sync_dual_tol_(true)
    , mu_init(mu_init)
    , verbose_(verbose)
    , hess_approx_(hess_approx)
    , sa_strategy_(sa_strategy)
    , max_iters(max_iters)
    , filter_(0.0, ls_params.alpha_min, ls_params.max_num_steps)
    , linesearch_() {
  ls_params.interp_type = LSInterpolation::CUBIC;
}

template <typename Scalar>
void SolverProxDDPTpl<Scalar>::setNumThreads(const size_t num_threads) {
  if (linear_solver_) {
    ALIGATOR_WARNING(
        "SolverProxDDP",
        "Linear solver already set: setNumThreads() should be called before "
        "you call setup() if you want to use the parallel linear solver.\n");
  }
  num_threads_ = num_threads;
  omp::set_default_options(num_threads);
}
// [1] Section IV. Proximal Differential Dynamic Programming
// C. Forward pass
template <typename Scalar>
Scalar SolverProxDDPTpl<Scalar>::tryLinearStep(const Problem &problem,
                                               const Scalar alpha) {
  ALIGATOR_TRACY_ZONE_SCOPED;

  const size_t nsteps = workspace_.nsteps;
  assert(results_.xs.size() == nsteps + 1);
  assert(results_.us.size() == nsteps);
  assert(results_.lams.size() == nsteps + 1);
  assert(results_.vs.size() == nsteps + 1);

  math::vectorMultiplyAdd(results_.lams, workspace_.dlams,
                          workspace_.trial_lams, alpha);
  math::vectorMultiplyAdd(results_.vs, workspace_.dvs, workspace_.trial_vs,
                          alpha);

  long ndx_max = 0U;
  long nu_max = 0U;
  for (size_t i = 0; i < nsteps; i++) {
    ndx_max = std::max(ndx_max, workspace_.dxs[i].size());
    nu_max = std::max(nu_max, workspace_.dus[i].size());
  }
  ndx_max = std::max(ndx_max, workspace_.dxs.back().size());
  VectorXs dx_tmp(ndx_max);
  VectorXs du_tmp(nu_max);

  for (size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];
    const int ndx = stage.ndx1();
    const int nu = stage.nu();
    dx_tmp.head(ndx) = alpha * workspace_.dxs[i];
    du_tmp.head(nu) = alpha * workspace_.dus[i];
    stage.xspace_->integrate(results_.xs[i], dx_tmp.head(ndx),
                             workspace_.trial_xs[i]);
    stage.uspace_->integrate(results_.us[i], du_tmp.head(nu),
                             workspace_.trial_us[i]);
  }
  const StageModel &stage = *problem.stages_[nsteps - 1];
  const long ndxN = workspace_.dxs[nsteps].size();
  dx_tmp.head(ndxN) = alpha * workspace_.dxs[nsteps];
  stage.xspace_next_->integrate(results_.xs[nsteps], dx_tmp.head(ndxN),
                                workspace_.trial_xs[nsteps]);
  TrajOptData &prob_data = workspace_.problem_data;
  prob_data.cost_ =
      problem.evaluate(workspace_.trial_xs, workspace_.trial_us, prob_data);
  return prob_data.cost_;
}

template <typename Scalar>
void SolverProxDDPTpl<Scalar>::setup(const Problem &problem) {
  if (!problem.checkIntegrity())
    ALIGATOR_RUNTIME_ERROR("Problem failed integrity check.");

  if (int(sa_strategy_) <= 1) {
    linesearch_.init(sa_strategy_, ls_params);
  }

  results_ = Results(problem);
  workspace_ = Workspace(problem);

  switch (linear_solver_choice) {
  case LQSolverChoice::SERIAL: {
    linear_solver_ = std::make_unique<gar::ProximalRiccatiSolver<Scalar>>(
        workspace_.lqr_problem);
    break;
  }
  case LQSolverChoice::PARALLEL: {
    if (rollout_type_ == RolloutType::NONLINEAR) {
      ALIGATOR_RUNTIME_ERROR(
          "Nonlinear rollouts not supported with the parallel solver.");
    }
#ifndef ALIGATOR_MULTITHREADING
    ALIGATOR_RUNTIME_ERROR(
        "Aligator was not compiled with OpenMP support. The parallel Riccati "
        "solver is not available.");
#else
    linear_solver_ = std::make_unique<gar::ParallelRiccatiSolver<Scalar>>(
        workspace_.lqr_problem, num_threads_);
#endif
    break;
  case LQSolverChoice::STAGEDENSE:
    linear_solver_ = std::make_unique<gar::RiccatiSolverDense<Scalar>>(
        workspace_.lqr_problem);
    break;
  }
  }
  filter_.resetFilter(0.0, ls_params.alpha_min, ls_params.max_num_steps);
}

template <typename Scalar>
void SolverProxDDPTpl<Scalar>::cycleProblem(const Problem &problem,
                                            const shared_ptr<StageData> &data) {
  results_.cycleAppend(problem, problem.getInitState());
  const auto nsteps = workspace_.nsteps;
  workspace_.cycleAppend(problem, data);
  linear_solver_->cycleAppend(workspace_.lqr_problem.stages[nsteps - 1]);
}

template <typename... VArgs> bool is_nan_any(const VArgs &...args) {
  return (math::check_value(args) || ...);
}

#define RET_FALSE_IF_NAN(...)                                                  \
  if (is_nan_any(__VA_ARGS__))                                                 \
    return false;

template <typename Scalar>
bool SolverProxDDPTpl<Scalar>::computeMultipliers(
    const Problem &problem, const std::vector<VectorXs> &lams,
    const std::vector<VectorXs> &vs) {
  ALIGATOR_TRACY_ZONE_SCOPED;
  using BlkView = BlkMatrix<VectorRef, -1, 1>;

  const TrajOptData &prob_data = workspace_.problem_data;
  const size_t nsteps = workspace_.nsteps;

  // [1] Section B. Augmented Lagrangian methods eqn. 5a and 5b for x_plus
  // and in subsection Primal-dual search x_k is prev_x. Then, more precisely
  // eqn. 39 gives the formula for first-order multiplier estimates in the
  // DDP setup
  std::vector<VectorXs> &lams_plus = workspace_.lams_plus;

  const std::vector<VectorXs> &vs_prev = workspace_.prev_vs;
  std::vector<VectorXs> &vs_plus = workspace_.vs_plus;
  // std::vector<VectorXs> &vs_pdal = workspace_.vs_pdal;

  std::vector<VectorXs> &Lvs = workspace_.Lvs;
  std::vector<VectorXs> &shifted_constraints = workspace_.shifted_constraints;

  assert(Lvs.size() == vs_prev.size());
  assert(Lvs.size() == nsteps + 1);

  // initial constraint
  {
    StageFunctionData &dd = *prob_data.init_data;
    lams_plus[0] = lams[0] + dd.value_ / mu();
    /// TODO: generalize to the other types of initial constraint (non-equality)
    workspace_.dyn_slacks[0] = dd.value_;
    RET_FALSE_IF_NAN(lams_plus[0]);
  }
  using ConstraintSetProd = ConstraintSetProductTpl<Scalar>;

  // loop over the stages
  for (size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];
    const StageData &sd = *prob_data.stage_data[i];
    const DynamicsData &dd = *sd.dynamics_data;
    const ConstraintStack &cstr_stack = stage.constraints_;

    assert(vs[i].size() == stage.nc());
    assert(lams[i + 1].size() == stage.ndx2());

    // 1. compute dynamics error
    lams_plus[i + 1] = lams[i + 1] + dd.value_ / mu();
    workspace_.dyn_slacks[i + 1] = dd.value_;
    RET_FALSE_IF_NAN(lams_plus[i + 1]);

    // 2. use product constraint operator
    // to compute the new multiplier estimates
    const ConstraintSetProd &op = workspace_.cstr_product_sets[i];

    // fill in shifted constraints buffer
    BlkView scvView(shifted_constraints[i], cstr_stack.dims());
    for (size_t j = 0; j < cstr_stack.size(); j++) {
      const StageFunctionData &cd = *sd.constraint_data[j];
      scvView[j] = cd.value_;
    }
    shifted_constraints[i] += mu() * vs_prev[i];
    op.normalConeProjection(shifted_constraints[i], vs_plus[i]);
    op.computeActiveSet(shifted_constraints[i],
                        workspace_.active_constraints[i]);
    Lvs[i] = vs_plus[i];
    Lvs[i].noalias() -= mu() * vs[i];
    vs_plus[i] = mu_inv() * vs_plus[i];
    assert(Lvs[i].size() == stage.nc());
    RET_FALSE_IF_NAN(Lvs[i]);
  }

  if (!problem.term_cstrs_.empty()) {
    assert(problem.term_cstrs_.size() == prob_data.term_cstr_data.size());
    const ConstraintStack &cstr_stack = problem.term_cstrs_;
    const ConstraintSetProd &op = workspace_.cstr_product_sets[nsteps];

    BlkView scvView(shifted_constraints[nsteps], cstr_stack.dims());
    for (size_t j = 0; j < cstr_stack.size(); j++) {
      const StageFunctionData &cd = *prob_data.term_cstr_data[j];
      scvView[j] = cd.value_;
    }
    shifted_constraints[nsteps] += mu() * vs_prev[nsteps];
    op.normalConeProjection(shifted_constraints[nsteps], vs_plus[nsteps]);
    op.computeActiveSet(shifted_constraints[nsteps],
                        workspace_.active_constraints[nsteps]);
    Lvs[nsteps] = vs_plus[nsteps];
    Lvs[nsteps].noalias() -= mu() * vs[nsteps];
    vs_plus[nsteps] = mu_inv() * vs_plus[nsteps];
    assert(Lvs[nsteps].size() == cstr_stack.totalDim());
    RET_FALSE_IF_NAN(Lvs[nsteps]);
  }
  return true;
}

#undef RET_FALSE_IF_NAN

template <typename Scalar> void SolverProxDDPTpl<Scalar>::updateGains() {
  ALIGATOR_TRACY_ZONE_SCOPED;
  ALIGATOR_NOMALLOC_SCOPED;
  using gar::StageFactor;
  const size_t N = workspace_.nsteps;
  linear_solver_->collapseFeedback(); // will alter feedback gains
  for (size_t i = 0; i < N; i++) {
    VectorRef ff = results_.getFeedforward(i);
    MatrixRef fb = results_.getFeedback(i);

    ff = linear_solver_->getFeedforward(i);
    fb = linear_solver_->getFeedback(i);
  }
  VectorRef ff = results_.getFeedforward(N);
  MatrixRef fb = results_.getFeedback(N);

  ff = linear_solver_->getFeedforward(N).tail(ff.rows());
  fb = linear_solver_->getFeedback(N).bottomRows(fb.rows());
}

// [1] Section IV. Proximal Differential Dynamic Programming
// C. Forward pass
template <typename Scalar>
Scalar SolverProxDDPTpl<Scalar>::tryNonlinearRollout(const Problem &problem,
                                                     const Scalar alpha) {
  ALIGATOR_TRACY_ZONE_SCOPED;
  using gar::StageFactor;

  const size_t nsteps = workspace_.nsteps;
  std::vector<VectorXs> &xs = workspace_.trial_xs;
  std::vector<VectorXs> &us = workspace_.trial_us;
  std::vector<VectorXs> &vs = workspace_.trial_vs;
  std::vector<VectorXs> &lams = workspace_.trial_lams;
  std::vector<VectorXs> &dxs = workspace_.dxs;
  std::vector<VectorXs> &dus = workspace_.dus;
  std::vector<VectorXs> &dvs = workspace_.dvs;
  std::vector<VectorXs> &dlams = workspace_.dlams;

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

  for (size_t t = 0; t < nsteps; t++) {
    const StageModel &stage = *problem.stages_[t];
    StageData &data = *prob_data.stage_data[t];

    const std::array<long, 4> _dims{stage.nu(), stage.nc(), stage.ndx2(),
                                    stage.ndx2()};
    BlkMatrix<ConstVectorRef, 4, 1> ff{
        linear_solver_->getFeedforward(t), _dims, {1}};
    BlkMatrix<ConstMatrixRef, 4, 1> fb{
        linear_solver_->getFeedback(t), _dims, {stage.ndx1()}};
    ConstVectorRef kff = ff[0];
    ConstVectorRef zff = ff[1];
    ConstVectorRef lff = ff[2];
    ConstMatrixRef Kfb = fb.blockRow(0);
    ConstMatrixRef Zfb = fb.blockRow(1);
    ConstMatrixRef Lfb = fb.blockRow(2);

    dus[t] = alpha * kff;
    dus[t].noalias() += Kfb * dxs[t];
    stage.uspace().integrate(results_.us[t], dus[t], us[t]);

    dvs[t] = alpha * zff;
    dvs[t].noalias() += Zfb * dxs[t];
    vs[t] = results_.vs[t] + dvs[t];

    dlams[t + 1] = alpha * lff;
    dlams[t + 1].noalias() += Lfb * dxs[t];
    lams[t + 1] = results_.lams[t + 1] + dlams[t + 1];

    stage.evaluate(xs[t], us[t], xs[t + 1], data);

    // rollout has zeroed out the residual
    dyn_slacks[t].setZero();

    if (!stage.hasDynModel() || stage.dynamics_->isExplicit()) {
      // nothing to do
    } else {
      ALIGATOR_RUNTIME_ERROR(
          "Implicit dynamics are no longer supported (for now)");
    }

    stage.xspace_next().difference(results_.xs[t + 1], xs[t + 1], dxs[t + 1]);

    ALIGATOR_RAISE_IF_NAN_NAME(xs[t + 1], fmt::format("xs[{:d}]", t + 1));
    ALIGATOR_RAISE_IF_NAN_NAME(us[t], fmt::format("us[{:d}]", t));
    ALIGATOR_RAISE_IF_NAN_NAME(lams[t + 1], fmt::format("lams[{:d}]", t + 1));
  }

  // TERMINAL NODE
  problem.term_cost_->evaluate(xs[nsteps], problem.unone_,
                               *prob_data.term_cost_data);

  for (size_t k = 0; k < problem.term_cstrs_.size(); ++k) {
    const auto &func = problem.term_cstrs_.funcs[k];
    StageFunctionData &td = *prob_data.term_cstr_data[k];
    func->evaluate(xs[nsteps], problem.unone_, td);
  }

  // update multiplier
  if (!problem.term_cstrs_.empty()) {
    const std::array<long, 2> _dims{
        workspace_.lqr_problem.stages[nsteps].nu,
        workspace_.lqr_problem.stages[nsteps].nc,
    };
    const uint ndx = workspace_.lqr_problem.stages[nsteps].nx;
    BlkMatrix<ConstVectorRef, 2, 1> ff{
        linear_solver_->getFeedforward(nsteps), _dims, {1}};
    BlkMatrix<ConstMatrixRef, 2, 1> fb{
        linear_solver_->getFeedback(nsteps), _dims, {ndx}};
    ConstVectorRef zff = ff[1];
    ConstMatrixRef Zfb = fb.blockRow(1);

    dvs[nsteps] = alpha * zff;
    dvs[nsteps].noalias() += Zfb * dxs[nsteps];
    vs[nsteps] = results_.vs[nsteps] + dvs[nsteps];
  }

  prob_data.cost_ = computeTrajectoryCost(prob_data);
  return prob_data.cost_;
}

// Main loop of the Algorithm 2 detailed in section II. Background
// on nonlinear optimization and augmented Lagrangian methods
template <typename Scalar>
bool SolverProxDDPTpl<Scalar>::run(const Problem &problem,
                                   const std::vector<VectorXs> &xs_init,
                                   const std::vector<VectorXs> &us_init,
                                   const std::vector<VectorXs> &vs_init,
                                   const std::vector<VectorXs> &lams_init) {

  if (sync_dual_tol_)
    target_dual_tol_ = target_tol_;

  ALIGATOR_TRACY_ZONE_SCOPED;
  if (!workspace_.isInitialized() || !results_.isInitialized()) {
    ALIGATOR_RUNTIME_ERROR("workspace and results were not allocated yet!");
  }
  if (mu_init < bcl_params.mu_lower_bound) {
    ALIGATOR_WARNING("SolverProxDDP",
                     "Initial value of mu_init < mu_lower_bound ({:.3g})\n",
                     bcl_params.mu_lower_bound);
    setAlmPenalty(mu_init);
  }

  detail::check_initial_guess_and_assign(problem, xs_init, us_init, results_.xs,
                                         results_.us);
  const bool v_was_resized =
      !vs_init.empty() && !assign_no_resize(vs_init, results_.vs);
  const bool l_was_resized =
      !lams_init.empty() && !assign_no_resize(lams_init, results_.lams);
  if (v_was_resized || l_was_resized) {
    ALIGATOR_WARNING("SolverProxDDP",
                     "Resize happened when initializing multipliers.\n");
  }

  if (force_initial_condition_) {
    workspace_.trial_xs[0] = problem.getInitState();
    workspace_.trial_lams[0].setZero();
  }

  logger.active = (verbose_ > 0);
  for (size_t j = 0; j < std::size(BASIC_KEYS); j++) {
    logger.addColumn(BASIC_KEYS[j]);
  }
  logger.printHeadline();

  setAlmPenalty(mu_init);

  workspace_.prev_xs = results_.xs;
  workspace_.prev_us = results_.us;
  workspace_.prev_vs = results_.vs;

  inner_tol_ = inner_tol0;
  prim_tol_ = prim_tol0;
  updateTolsOnFailure();

  inner_tol_ = std::max(inner_tol_, target_dual_tol_);
  prim_tol_ = std::max(prim_tol_, target_tol_);

  bool &conv = results_.conv = false;

  results_.al_iter = 0;
  results_.num_iters = 0;
  size_t &al_iter = results_.al_iter;
  while ((al_iter < max_al_iters) && (results_.num_iters < max_iters)) {
    if (!innerLoop(problem)) {
      al_iter++;
      break;
    }
    linesearch_.reset();

    // accept primal updates
    workspace_.prev_xs = results_.xs;
    workspace_.prev_us = results_.us;

    if (results_.prim_infeas <= prim_tol_) {
      do {
        updateTolsOnSuccess();
      } while (workspace_.inner_criterion < inner_tol_);

      switch (multiplier_update_mode) {
      case MultiplierUpdateMode::NEWTON:
        workspace_.prev_vs = results_.vs;
        break;
      case MultiplierUpdateMode::PRIMAL:
        workspace_.prev_vs = workspace_.vs_plus;
        break;
      case MultiplierUpdateMode::PRIMAL_DUAL:
        workspace_.prev_vs = workspace_.vs_pdal;
        break;
      default:
        break;
      }

      if ((results_.dual_infeas <= target_dual_tol_) &&
          (results_.prim_infeas <= target_tol_)) {
        conv = true;
        break;
      }
    } else {
      setAlmPenalty(mu_penal_ * bcl_params.mu_update_factor);
      updateTolsOnFailure();
      if (math::scalar_close(mu_penal_, bcl_params.mu_lower_bound)) {
        // reset penalty to initial value
        setAlmPenalty(mu_init);
      }
    }

    inner_tol_ = std::max(inner_tol_, 0.01 * target_dual_tol_);
    prim_tol_ = std::max(prim_tol_, target_tol_);

    al_iter++;
  }

  logger.finish(conv);
  return conv;
}

// [1] Algorithm 2, l. 5: this is the function the linesearch
// is performed on.
template <typename Scalar>
Scalar SolverProxDDPTpl<Scalar>::forwardPass(const Problem &problem,
                                             const Scalar alpha) {
  ALIGATOR_TRACY_ZONE_SCOPED;
  switch (rollout_type_) {
  case RolloutType::LINEAR:
    tryLinearStep(problem, alpha);
    break;
  case RolloutType::NONLINEAR:
    tryNonlinearRollout(problem, alpha);
    break;
  }
  if (!computeMultipliers(problem, workspace_.trial_lams, workspace_.trial_vs))
    ALIGATOR_RUNTIME_ERROR(
        "computeMultipliers() returned false. NaN or Inf detected.");
  return ALFunction<Scalar>::evaluate(mu(), mu(), problem, workspace_.lams_plus,
                                      workspace_.trial_vs, workspace_);
}

template <typename Scalar>
bool SolverProxDDPTpl<Scalar>::innerLoop(const Problem &problem) {
  ALIGATOR_TRACY_ZONE_NAMED(InnerLoop, true);

  auto merit_eval_fun = [&](Scalar a0) -> Scalar {
    return forwardPass(problem, a0);
  };

  auto pair_eval_fun = [&](Scalar a0) -> std::pair<Scalar, Scalar> {
    ALIGATOR_TRACY_ZONE_NAMED_N(FilterPairEval, "pair_eval_fun", true);
    std::pair<Scalar, Scalar> fpair;
    fpair.first = forwardPass(problem, a0);
    computeInfeasibilities(problem);
    fpair.second = results_.prim_infeas;
    return fpair;
  };

  size_t &iter = results_.num_iters;
  results_.traj_cost_ = problem.evaluate(results_.xs, results_.us,
                                         workspace_.problem_data, num_threads_);
  computeMultipliers(problem, results_.lams, results_.vs);
  results_.merit_value_ = ALFunction<Scalar>::evaluate(
      mu(), mu(), problem, workspace_.lams_plus, results_.vs, workspace_);

  for (; iter < max_iters; iter++) {
    ALIGATOR_TRACY_ZONE_NAMED_N(ZoneIteration, "inner_iteration", true);
    // ASSUMPTION: last evaluation in previous iterate
    // was during linesearch, at the current candidate solution (x,u).
    /// TODO: make this smarter using e.g. some caching mechanism
    problem.computeDerivatives(results_.xs, results_.us,
                               workspace_.problem_data, num_threads_);
    const Scalar phi0 = results_.merit_value_;

    // compute the Lagrangian derivatives to check for convergence
    // and use them in the LQ subproblem as gradient of cost g
    // with cstr_lx_corr.
    LagrangianDerivatives<Scalar>::compute(problem, workspace_.problem_data,
                                           results_.lams, results_.vs,
                                           workspace_.Lxs, workspace_.Lus);
    if (force_initial_condition_) {
      workspace_.Lxs[0].setZero();
    }
    computeInfeasibilities(problem);
    computeCriterion();

    // exit if either the subproblem or overall problem converged
    const bool overall_converged = (results_.dual_infeas <= target_dual_tol_) &&
                                   (results_.prim_infeas <= target_tol_);
    if ((workspace_.inner_criterion <= inner_tol_) || overall_converged)
      return true;

    computeProjectedJacobians(problem, mu_inv(), workspace_);
    this->initializeRegularization();
    this->updateLQSubproblem();
    // TODO: supply a penalty weight matrix for constraints

    // In the next two lines, the LQ subproblem is solved. This is
    // another way to view the backward and forward passes of the
    // Riccati recursion detailed in [1] (IV. Proximal Differential
    //  Dynamic Programming Section B Backward). The backward pass
    // computes the gains, and the forward pass computes the new
    // control and state trajectories.
    linear_solver_->backward(mu());

    linear_solver_->forward(workspace_.dxs, workspace_.dus, workspace_.dvs,
                            workspace_.dlams);
    this->updateGains();

    if (force_initial_condition_) {
      workspace_.dxs[0].setZero();
      workspace_.dlams[0].setZero();
    }
    Scalar dphi0 = ALFunction<Scalar>::directionalDerivative(
        mu(), mu(), problem, results_.lams, results_.vs, workspace_);
    ALIGATOR_RAISE_IF_NAN(dphi0);

    // check if we can early stop
    if (std::abs(dphi0) <= ls_params.dphi_thresh)
      return true;

    // otherwise continue linesearch
    Scalar alpha_opt = 1;
    Scalar phi_new;

    switch (sa_strategy_) {
    case StepAcceptanceStrategy::LINESEARCH_ARMIJO:
    case StepAcceptanceStrategy::LINESEARCH_NONMONOTONE:
      assert(linesearch_.isValid());
      phi_new = linesearch_.run(merit_eval_fun, phi0, dphi0, alpha_opt);
      break;
    case StepAcceptanceStrategy::FILTER:
      phi_new = filter_.run(pair_eval_fun, alpha_opt);
      break;
    default:
      assert(false && "unknown StepAcceptanceStrategy!");
      break;
    }

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

    if (iter >= 1 && iter % 25 == 0)
      logger.printHeadline();
    logger.addEntry("iter", iter + 1);
    logger.addEntry("alpha", alpha_opt);
    logger.addEntry("inner_crit", workspace_.inner_criterion);
    logger.addEntry("prim_err", results_.prim_infeas);
    logger.addEntry("dual_err", results_.dual_infeas);
    logger.addEntry("preg", preg_);
    logger.addEntry("dphi0", dphi0);
    logger.addEntry("cost", results_.traj_cost_);
    logger.addEntry("merit", phi_new);
    logger.addEntry("ΔM", phi_new - phi0);
    logger.addEntry("aliter", results_.al_iter + 1);
    logger.addEntry("mu", mu());

    if (alpha_opt <= ls_params.alpha_min) {
      if (preg_ >= reg_max)
        return false;
      increaseRegularization();
    }
    this->invokeCallbacks();
    logger.log();

    preg_last_ = preg_;
  }
  return false;
}

template <typename Scalar>
void SolverProxDDPTpl<Scalar>::computeInfeasibilities(const Problem &problem) {
  ALIGATOR_NOMALLOC_SCOPED;
  ALIGATOR_TRACY_ZONE_SCOPED;
  const size_t nsteps = workspace_.nsteps;

  std::vector<VectorXs> &vs_plus = workspace_.vs_plus;
  std::vector<VectorXs> &vs_prev = workspace_.prev_vs;
  std::vector<VectorXs> &stage_infeas = workspace_.stage_infeasibilities;

  // compute infeasibility of all stage constraints [1] eqn. 53
  for (size_t i = 0; i < nsteps; i++) {
    stage_infeas[i] = mu() * (vs_plus[i] - vs_prev[i]);

    workspace_.stage_cstr_violations[long(i)] =
        math::infty_norm(stage_infeas[i]);
  }

  // compute infeasibility of terminal constraints
  if (!problem.term_cstrs_.empty()) {
    stage_infeas[nsteps] = mu() * (vs_plus[nsteps] - vs_prev[nsteps]);

    workspace_.stage_cstr_violations[long(nsteps)] =
        math::infty_norm(stage_infeas[nsteps]);
  }

  results_.prim_infeas = std::max(math::infty_norm(stage_infeas),
                                  math::infty_norm(workspace_.dyn_slacks));
}

template <typename Scalar> void SolverProxDDPTpl<Scalar>::computeCriterion() {
  ALIGATOR_NOMALLOC_SCOPED;
  ALIGATOR_TRACY_ZONE_SCOPED;
  const size_t nsteps = workspace_.nsteps;

  workspace_.stage_inner_crits.setZero();

  for (size_t i = 0; i < nsteps; i++) {
    // dual residual
    Scalar rx = math::infty_norm(workspace_.Lxs[i]);
    Scalar ru = math::infty_norm(workspace_.Lus[i]);
    // dynamics residual
    Scalar rd = math::infty_norm(workspace_.dyn_slacks[i]);
    // path constraints
    Scalar rc = math::infty_norm(workspace_.Lvs[i]);

    workspace_.stage_inner_crits[long(i)] = std::max({rx, ru, rd, rc});
    workspace_.state_dual_infeas[long(i)] = rx; // [1] eqn. 52
    workspace_.control_dual_infeas[long(i)] = ru;
  }
  Scalar rx = math::infty_norm(workspace_.Lxs[nsteps]);
  Scalar rc = math::infty_norm(workspace_.Lvs[nsteps]);
  workspace_.state_dual_infeas[long(nsteps)] = rx; // [1] eqn. 52
  workspace_.stage_inner_crits[long(nsteps)] = std::max(rx, rc);

  workspace_.inner_criterion = math::infty_norm(workspace_.stage_inner_crits);
  results_.dual_infeas =
      std::max(math::infty_norm(workspace_.state_dual_infeas),
               math::infty_norm(workspace_.control_dual_infeas));
}

template <typename Scalar>
void SolverProxDDPTpl<Scalar>::registerCallback(const std::string &name,
                                                CallbackPtr cb) {
  callbacks_.insert_or_assign(name, cb);
}

template <typename Scalar> void SolverProxDDPTpl<Scalar>::updateLQSubproblem() {
  ALIGATOR_NOMALLOC_SCOPED;
  ALIGATOR_TRACY_ZONE_SCOPED;
  gar::LqrProblemTpl<Scalar> &prob = workspace_.lqr_problem;
  const TrajOptData &pd = workspace_.problem_data;

  using gar::LqrKnotTpl;

  size_t N = (size_t)prob.horizon();
  assert(N == workspace_.nsteps);

  for (size_t t = 0; t < N; t++) {
    const StageData &sd = *pd.stage_data[t];
    // taking the knot and not a view, since resizes might
    // happen here.
    LqrKnotTpl<Scalar> &knot = prob.stages[t];
    const DynamicsData &dd = *sd.dynamics_data;
    const CostData &cd = *sd.cost_data;
    uint nx = knot.nx;
    uint nu = knot.nu;
    uint nc = knot.nc;

    knot.A = dd.Jx_;
    knot.B = dd.Ju_;
    knot.f = dd.value_;

    knot.Q = cd.Lxx_;
    knot.S = cd.Lxu_;
    knot.R = cd.Luu_;
    knot.q = workspace_.Lxs[t];
    knot.r = workspace_.Lus[t];

    knot.Q.diagonal().array() += preg_;
    knot.R.diagonal().array() += preg_;

    // dynamics hessians
    if (hess_approx_ == HessianApprox::EXACT) {
      knot.Q += dd.Hxx_;
      knot.S += dd.Hxu_;
      knot.R += dd.Huu_;
    }

    // TODO: handle the bloody constraints
    assert(knot.nc == workspace_.cstr_proj_jacs[t].rows());
    knot.C.topRows(nc) = workspace_.cstr_proj_jacs[t].blockCol(0);
    knot.D.topRows(nc) = workspace_.cstr_proj_jacs[t].blockCol(1);
    knot.d.head(nc) = workspace_.Lvs[t];

    // correct right-hand side
    knot.q.head(nx) += workspace_.cstr_lx_corr[t];
    knot.r.head(nu) += workspace_.cstr_lu_corr[t];
  }

  {
    LqrKnotTpl<Scalar> &knot = prob.stages[N];
    const CostData &tcd = *pd.term_cost_data;
    knot.Q = tcd.Lxx_;
    knot.Q.diagonal().array() += preg_;
    knot.q = workspace_.Lxs[N];
    knot.C = workspace_.cstr_proj_jacs[N].blockCol(0);
    knot.d = workspace_.Lvs[N];
    // correct right-hand side
    knot.q += workspace_.cstr_lx_corr[N];
  }

  const StageFunctionData &id = *pd.init_data;
  prob.G0 = id.Jx_;
  prob.g0 = id.value_;

  LqrKnotTpl<Scalar> &model = prob.stages[0];
  model.Q += id.Hxx_;
}

} // namespace aligator
