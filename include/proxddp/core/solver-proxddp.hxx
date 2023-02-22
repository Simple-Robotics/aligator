/// @file solver-proxddp.hxx
/// @brief  Implementations for the trajectory optimization algorithm.
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/solver-proxddp.hpp"
#include "proxddp/core/linalg.hpp"
#ifndef NDEBUG
#include <fmt/ostream.h>
#endif

namespace proxddp {

template <typename Scalar>
SolverProxDDP<Scalar>::SolverProxDDP(const Scalar tol, const Scalar mu_init,
                                     const Scalar rho_init,
                                     const std::size_t max_iters,
                                     VerboseLevel verbose,
                                     HessianApprox hess_approx)
    : target_tol_(tol), mu_init(mu_init), rho_init(rho_init), verbose_(verbose),
      hess_approx_(hess_approx), rollout_type_(RolloutType::NONLINEAR),
      is_x0_fixed_(true), ldlt_algo_choice_(LDLTChoice::DENSE),
      max_iters(max_iters), linesearch_(ls_params) {
  ls_params.interp_type = proxnlp::LSInterpolation::CUBIC;
}

template <typename Scalar>
void SolverProxDDP<Scalar>::linearRollout(const Problem &problem,
                                          Workspace &workspace,
                                          const Results &results) const {
  PROXDDP_NOMALLOC_BEGIN;
  compute_dir_x0(problem, workspace, results);

  const std::size_t nsteps = workspace.nsteps;

  for (std::size_t i = 0; i < nsteps; i++) {
    VectorXs &pd_step = workspace.pd_step_[i + 1];
    const auto ff = results.getFeedforward(i);
    const auto fb = results.getFeedback(i);

    pd_step = ff;
    pd_step.noalias() += fb * workspace.dxs[i];
  }
  if (!problem.term_cstrs_.empty()) {
    const auto ff = results.getFeedforward(nsteps);
    const auto fb = results.getFeedback(nsteps);
    VectorRef &dlam = workspace.dlams.back();
    dlam = ff;
    dlam.noalias() += fb * workspace.dxs[nsteps];
  }
  PROXDDP_NOMALLOC_END;
}

template <typename Scalar>
Scalar SolverProxDDP<Scalar>::forward_linear_impl(const Problem &problem,
                                                  Workspace &workspace,
                                                  const Results &results,
                                                  const Scalar alpha) const {

  const std::size_t nsteps = workspace.nsteps;

  for (std::size_t i = 0; i < results.lams.size(); i++) {
    workspace.trial_lams[i] = results.lams[i];
    workspace.trial_lams[i] += alpha * workspace.dlams[i];
  }

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
void SolverProxDDP<Scalar>::compute_dir_x0(const Problem &problem,
                                           Workspace &workspace,
                                           const Results &results) const {
  PROXDDP_NOMALLOC_BEGIN;
  // compute direction dx0
  const VParams &vp = workspace.value_params[0];
  const FunctionData &init_data = workspace.problem_data.getInitData();
  const int ndual0 = problem.init_state_error_->nr;
  const int ndx0 = problem.init_state_error_->ndx1;
  const VectorXs &lampl0 = workspace.lams_plus[0];
  const VectorXs &lamin0 = results.lams[0];
  const CostData &proxdata0 = *workspace.prox_datas[0];
  MatrixXs &kkt_mat = workspace.kkt_mats_[0];
  VectorRef kkt_rhs = workspace.kkt_rhs_[0].col(0);
  VectorRef kktx = kkt_rhs.head(ndx0);
  assert(kkt_rhs.size() == ndx0 + ndual0);
  assert(kkt_mat.cols() == ndx0 + ndual0);

  if (is_x0_fixed_) {
    workspace.pd_step_[0].setZero();
    workspace.trial_lams[0].setZero();
    kkt_rhs.setZero();

  } else {
    auto kktl = kkt_rhs.tail(ndual0);
    kktx = vp.Vx_;
    kktx.noalias() += init_data.Jx_.transpose() * lamin0;
    kktl = mu() * (lampl0 - lamin0);

    auto kkt_xx = kkt_mat.topLeftCorner(ndx0, ndx0);
    kkt_xx = vp.Vxx_ + init_data.Hxx_;
    if (rho() > 0)
      kkt_xx += rho() * proxdata0.Lxx_;

    kkt_mat.topRightCorner(ndx0, ndual0) = init_data.Jx_.transpose();
    kkt_mat.bottomLeftCorner(ndual0, ndx0) = init_data.Jx_;
    kkt_mat.bottomRightCorner(ndual0, ndual0).diagonal().array() = -mu();
    typename Workspace::LDLT &ldlt = *workspace.ldlts_[0];
    PROXDDP_NOMALLOC_END;
    ldlt.compute(kkt_mat);
    iterative_refinement_impl<Scalar>::run(
        ldlt, kkt_mat, kkt_rhs, workspace.kkt_resdls_[0], workspace.pd_step_[0],
        refinement_threshold_, max_refinement_steps_);
    PROXDDP_NOMALLOC_BEGIN;
  }
  PROXDDP_NOMALLOC_END;
}

template <typename Scalar>
void SolverProxDDP<Scalar>::setup(const Problem &problem) {
  workspace_ = std::make_unique<Workspace>(problem, ldlt_algo_choice_);
  results_ = std::make_unique<Results>(problem);
  linesearch_.setOptions(ls_params);

  Workspace &ws = *workspace_;
  prox_penalties_.clear();
  const std::size_t nsteps = ws.nsteps;
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &sm = *problem.stages_[i];
    prox_penalties_.emplace_back(sm.xspace_, sm.uspace_, ws.prev_xs[i],
                                 ws.prev_us[i], false);
    if (i == nsteps - 1) {
      prox_penalties_.emplace_back(sm.xspace_next_, sm.uspace_,
                                   ws.prev_xs[nsteps], problem.unone_, true);
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
void SolverProxDDP<Scalar>::computeProxTerms(const std::vector<VectorXs> &xs,
                                             const std::vector<VectorXs> &us,
                                             Workspace &workspace) const {
  const std::size_t nsteps = workspace.nsteps;
  for (std::size_t i = 0; i < nsteps; i++) {
    prox_penalties_[i].evaluate(xs[i], us[i], *workspace.prox_datas[i]);
  }
  prox_penalties_[nsteps].evaluate(xs[nsteps], us[nsteps - 1],
                                   *workspace.prox_datas[nsteps]);
}

template <typename Scalar>
void SolverProxDDP<Scalar>::computeProxDerivatives(
    const std::vector<VectorXs> &xs, const std::vector<VectorXs> &us,
    Workspace &workspace) const {
  const std::size_t nsteps = workspace.nsteps;
  for (std::size_t i = 0; i < nsteps; i++) {
    prox_penalties_[i].computeGradients(xs[i], us[i], *workspace.prox_datas[i]);
    prox_penalties_[i].computeHessians(xs[i], us[i], *workspace.prox_datas[i]);
  }
  prox_penalties_[nsteps].computeGradients(xs[nsteps], us[nsteps - 1],
                                           *workspace.prox_datas[nsteps]);
  prox_penalties_[nsteps].computeHessians(xs[nsteps], us[nsteps - 1],
                                          *workspace.prox_datas[nsteps]);
}

template <typename Scalar>
bool SolverProxDDP<Scalar>::backwardPass(const Problem &problem,
                                         Workspace &workspace,
                                         Results &results) const {
  /* Terminal node */
  computeTerminalValue(problem, workspace, results);

  const std::size_t nsteps = workspace.nsteps;
  for (std::size_t i = 0; i < nsteps; i++) {
    std::size_t t = nsteps - i - 1;
    updateHamiltonian(problem, results, workspace, t);
    bool b = computeGains(problem, workspace, results, t);
    if (!b) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
void SolverProxDDP<Scalar>::computeMultipliers(
    const Problem &problem, Workspace &workspace,
    const std::vector<VectorXs> &lams) const {

  TrajOptData &prob_data = workspace.problem_data;
  const std::size_t nsteps = workspace.nsteps;

  std::vector<VectorXs> &lams_prev = workspace.lams_prev;
  std::vector<VectorXs> &lams_plus = workspace.lams_plus;
  std::vector<VectorXs> &lams_pdal = workspace.lams_pdal;
  std::vector<VectorXs> &shifted_constraints = workspace.shifted_constraints;

  // initial constraint
  {
    const VectorXs &lam0 = lams[0];
    const VectorXs &plam0 = lams_prev[0];
    FunctionData &data = prob_data.getInitData();
    shifted_constraints[0] = data.value_ + mu() * plam0;
    lams_plus[0] = shifted_constraints[0] * mu_inv();
    lams_pdal[0] = (1 + dual_weight) * lams_plus[0] - dual_weight * lam0;
    /// TODO: generalize to the other types of initial constraint (non-equality)
  }

  using FuncDataVec = std::vector<shared_ptr<FunctionData>>;
  auto execute_on_stack =
      [dual_weight = dual_weight](
          const ConstraintStack &stack, const VectorXs &lambda,
          const VectorXs &prevlam, VectorXs &lamplus, VectorXs &lampdal,
          VectorXs &shift_cvals, const FuncDataVec &constraint_data,
          CstrALWeightStrat &&weights) {
        // k: constraint count variable
        for (std::size_t k = 0; k < stack.size(); k++) {
          const auto lami_k = stack.getConstSegmentByConstraint(lambda, k);
          const auto plami_k = stack.getConstSegmentByConstraint(prevlam, k);
          auto lamplus_k = stack.getSegmentByConstraint(lamplus, k);
          auto lampdal_k = stack.getSegmentByConstraint(lampdal, k);
          auto scval_k = stack.getSegmentByConstraint(shift_cvals, k);
          const CstrSet &set = stack.getConstraintSet(k);
          const FunctionData &data = *constraint_data[k];

          scval_k = data.value_ + weights.get(k) * plami_k;
          lamplus_k = scval_k;

          set.normalConeProjection(lamplus_k, lamplus_k);

          // set multiplier = 1/mu * normal_proj(shifted_cstr)
          lamplus_k *= weights.inv(k);
          lampdal_k = (1 + dual_weight) * lamplus_k - dual_weight * lami_k;
        }
      };

  // loop over the stages
#pragma omp parallel for num_threads(problem.getNumThreads())
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];
    const StageData &sdata = *prob_data.stage_data[i];
    const ConstraintStack &cstr_stack = stage.constraints_;

    const VectorXs &lami = lams[i + 1];
    const VectorXs &plami = lams_prev[i + 1];
    VectorXs &lamplusi = lams_plus[i + 1];
    VectorXs &lampdali = lams_pdal[i + 1];
    VectorXs &shiftcvali = shifted_constraints[i + 1];

    execute_on_stack(cstr_stack, lami, plami, lamplusi, lampdali, shiftcvali,
                     sdata.constraint_data, CstrALWeightStrat(mu_penal_, true));
  }

  if (!problem.term_cstrs_.empty()) {
    const VectorXs &lamN = lams.back();
    const VectorXs &plamN = lams_prev.back();
    VectorXs &lamplusN = lams_plus.back();
    VectorXs &lampdalN = lams_pdal.back();
    VectorXs &scval = shifted_constraints.back();
    execute_on_stack(problem.term_cstrs_, lamN, plamN, lamplusN, lampdalN,
                     scval, prob_data.term_cstr_data,
                     CstrALWeightStrat(mu_penal_, false));
  }
}

template <typename Scalar>
void SolverProxDDP<Scalar>::projectJacobians(const Problem &problem,
                                             Workspace &workspace) const {
  PROXDDP_NOMALLOC_BEGIN;
  TrajOptData &prob_data = workspace.problem_data;

  const std::size_t nsteps = workspace.nsteps;

  const std::vector<VectorXs> &shifted_constraints =
      workspace.shifted_constraints;

  using FuncDataVec = std::vector<shared_ptr<FunctionData>>;
  auto execute_on_stack = [](const ConstraintStack &stack,
                             const VectorXs &shift_cvals,
                             FuncDataVec &constraint_data) {
    for (std::size_t k = 0; k < stack.size(); ++k) {
      const auto scval_k = stack.getConstSegmentByConstraint(shift_cvals, k);
      const CstrSet &set = stack.getConstraintSet(k);
      FunctionData &data = *constraint_data[k];
      set.applyNormalConeProjectionJacobian(scval_k, data.jac_buffer_);
    }
  };

  // loop over the stages
#pragma omp parallel for num_threads(problem.getNumThreads())
  for (std::size_t i = 0; i < nsteps; i++) {
    const ConstraintStack &cstr_stack = problem.stages_[i]->constraints_;
    StageData &sdata = prob_data.getStageData(i);
    const VectorXs &shift_cval = shifted_constraints[i + 1];
    execute_on_stack(cstr_stack, shift_cval, sdata.constraint_data);
  }

  // terminal node
  if (!problem.term_cstrs_.empty()) {
    execute_on_stack(problem.term_cstrs_, shifted_constraints.back(),
                     prob_data.term_cstr_data);
  }
  PROXDDP_NOMALLOC_END;
}

template <typename Scalar>
void SolverProxDDP<Scalar>::updateHamiltonian(const Problem &problem,
                                              const Results &results,
                                              Workspace &workspace,
                                              const std::size_t t) const {
  PROXDDP_NOMALLOC_BEGIN;

  const StageModel &stage = *problem.stages_[t];

  const VParams &vnext = workspace.value_params[t + 1];
  QParams &qparam = workspace.q_params[t];

  StageData &stage_data = workspace.problem_data.getStageData(t);
  const CostData &cdata = *stage_data.cost_data;
  const CostData &proxdata = *workspace.prox_datas[t];

  const int ndx1 = stage.ndx1();
  const int nu = stage.nu();

  // Use the contiguous full gradient/jacobian/hessian buffers
  // to fill in the Q-function derivatives
  qparam.q_ = cdata.value_; // rho() * proxdata.value_;
  qparam.Qx = cdata.Lx_;    // rho() * proxdata.Lx_;
  qparam.Qu = cdata.Lu_;    //+ rho() * proxdata.Lu_;
  qparam.Qy = vnext.Vx_;

  auto qpar_xu = qparam.hess_.topLeftCorner(ndx1 + nu, ndx1 + nu);
  qpar_xu = cdata.hess_; //+ rho() * proxdata.hess_;
  qparam.Qyy = vnext.Vxx_;
  qparam.Quu.diagonal().array() += ureg_;

  const VectorXs &lam = results.lams[t + 1];
  if (rho() > 0) {
    qparam.q_ += rho() * proxdata.value_;
    qparam.Qx += rho() * proxdata.Lx_;
    qparam.Qu += rho() * proxdata.Lu_;
    qpar_xu += rho() * proxdata.hess_;
  }

  const ConstraintStack &cstr_stack = stage.constraints_;
  for (std::size_t k = 0; k < cstr_stack.size(); k++) {
    FunctionData &cstr_data = *stage_data.constraint_data[k];

    const auto lam_j = cstr_stack.getConstSegmentByConstraint(lam, k);
    qparam.grad_.noalias() += cstr_data.jac_buffer_.transpose() * lam_j;
    if (hess_approx_ == HessianApprox::EXACT) {
      qparam.hess_ += cstr_data.vhp_buffer_;
    }
  }
  PROXDDP_NOMALLOC_END;
}

template <typename Scalar>
void SolverProxDDP<Scalar>::computeTerminalValue(const Problem &problem,
                                                 Workspace &workspace,
                                                 Results &results) const {
  PROXDDP_NOMALLOC_BEGIN;
  const std::size_t nsteps = workspace.nsteps;

  const TrajOptData &prob_data = workspace.problem_data;
  const CostData &term_cost_data = *prob_data.term_cost_data;
  VParams &term_value = workspace.value_params[nsteps];
  const CostData &proxdata = *workspace.prox_datas[nsteps];

  term_value.v_ = term_cost_data.value_;
  term_value.Vx_ = term_cost_data.Lx_;
  term_value.Vxx_ = term_cost_data.Lxx_;
  term_value.Vxx_.diagonal().array() += xreg_;

  if (rho() > 0.) {
    term_value.v_ += rho() * proxdata.value_;
    term_value.Vx_ += rho() * proxdata.Lx_;
    term_value.Vxx_ += rho() * proxdata.Lxx_;
  }

  if (!problem.term_cstrs_.empty()) {
    /* check number of multipliers */
    assert(results.lams.size() == (nsteps + 2));
    assert(results.gains_.size() == (nsteps + 1));
  }

  for (std::size_t k = 0; k < problem.term_cstrs_.size(); ++k) {
    const FunctionData &cstr_data = *prob_data.term_cstr_data[k];

    const VectorXs &lamplus = workspace.lams_plus[nsteps + 1];
    const VectorXs &lamin = results.lams[nsteps + 1];
    const MatrixRef &constraint_Jx = cstr_data.Jx_;

    auto ff = results.getFeedforward(nsteps);
    auto fb = results.getFeedback(nsteps);
    ff = lamplus - lamin;
    fb = mu_inv() * constraint_Jx;

    term_value.v_ += 0.5 * mu_inv() * lamplus.squaredNorm();
    term_value.Vx_.noalias() += constraint_Jx.transpose() * lamplus;
    term_value.Vxx_ += cstr_data.Hxx_;
    term_value.Vxx_.noalias() += constraint_Jx.transpose() * fb;
  }
  PROXDDP_NOMALLOC_END;
}

template <typename Scalar>
bool SolverProxDDP<Scalar>::computeGains(const Problem &problem,
                                         Workspace &workspace, Results &results,
                                         const std::size_t t) const {
  PROXDDP_NOMALLOC_BEGIN;
  const StageModel &stage = *problem.stages_[t];

  const QParams &qparam = workspace.q_params[t];

  StageData &stage_data = workspace.problem_data.getStageData(t);

  const int nprim = stage.numPrimal();
  const int ndual = stage.numDual();
  const int ndx1 = stage.ndx1();
  const int nu = stage.nu();
  const int ndx2 = stage.ndx2();

  const VectorXs &laminnr = results.lams[t + 1];
  const VectorXs &lamplus = workspace.lams_plus[t + 1];

  MatrixXs &kkt_mat = workspace.kkt_mats_[t + 1];
  MatrixXs &kkt_rhs = workspace.kkt_rhs_[t + 1];

  assert(kkt_mat.rows() == (nprim + ndual));
  assert(kkt_rhs.rows() == (nprim + ndual));
  assert(kkt_rhs.cols() == (ndx1 + 1));

  BlockXs kkt_jac = kkt_mat.block(nprim, 0, ndual, nprim);
  BlockXs kkt_top_left = kkt_mat.topLeftCorner(nprim, nprim);
  Eigen::Diagonal<BlockXs> kkt_low_right =
      kkt_mat.bottomRightCorner(ndual, ndual).diagonal();

  typename MatrixXs::ColXpr kkt_rhs_ff = kkt_rhs.col(0);
  typename MatrixXs::ColsBlockXpr kkt_rhs_fb = kkt_rhs.rightCols(ndx1);

  // blocks: u, y, and dual
  kkt_rhs_ff.head(nu) = qparam.Qu;
  kkt_rhs_ff.segment(nu, ndx2) = qparam.Qy;

  kkt_rhs_fb.topRows(nu) = qparam.Qxu.transpose();
  kkt_rhs_fb.middleRows(nu, ndx2) = qparam.Qxy.transpose();

  // KKT matrix: (u, y)-block = bottom right of q hessian
  kkt_top_left = qparam.hess_.bottomRightCorner(nprim, nprim);

  const ConstraintStack &cstr_mgr = stage.constraints_;

  // Loop over constraints
  for (std::size_t j = 0; j < stage.numConstraints(); j++) {
    FunctionData &cstr_data = *stage_data.constraint_data[j];
    const auto laminnr_j = cstr_mgr.getConstSegmentByConstraint(laminnr, j);
    const auto lamplus_j = cstr_mgr.getConstSegmentByConstraint(lamplus, j);

    CstrALWeightStrat weight_strat(mu_penal_, true);
    // update the KKT jacobian columns
    cstr_mgr.getBlockByConstraint(kkt_jac, j) =
        cstr_data.jac_buffer_.rightCols(nprim);
    cstr_mgr.getBlockByConstraint(kkt_rhs_fb.bottomRows(ndual), j) =
        cstr_data.Jx_;
    cstr_mgr.getSegmentByConstraint(kkt_rhs_ff.tail(ndual), j) =
        weight_strat.get(j) * (lamplus_j - laminnr_j);

    kkt_low_right.array() = -weight_strat.get(j);
  }

  /* Compute gains with LDLT */
  kkt_mat = kkt_mat.template selfadjointView<Eigen::Lower>();
  typename Workspace::LDLT &ldlt = *workspace.ldlts_[t + 1];
  PROXDDP_NOMALLOC_END;
  ldlt.compute(kkt_mat);
  PROXDDP_NOMALLOC_BEGIN;

  // check inertia
  {
    PROXDDP_RAISE_IF_NAN_NAME(ldlt.vectorD(), "ldlt.vectorD()");
    math::compute_inertia(ldlt.vectorD(), workspace.inertia.data());
    if ((workspace.inertia[1] > 0U) ||
        (workspace.inertia[2] != (std::size_t)ndual)) {
      return false;
    }
  }

  MatrixXs &gains = results.gains_[t];
  MatrixXs &resdl = workspace.kkt_resdls_[t + 1];

  PROXDDP_NOMALLOC_END;
  iterative_refinement_impl<Scalar>::run(ldlt, kkt_mat, kkt_rhs, resdl, gains,
                                         refinement_threshold_,
                                         max_refinement_steps_);
  PROXDDP_NOMALLOC_BEGIN;

  /* Value function */
  VParams &vp = workspace.value_params[t];
  auto Qxw = kkt_rhs_fb.transpose();
  auto ff = results.getFeedforward(t);
  auto fb = results.getFeedback(t);

  vp.Vx_ = qparam.Qx;
  vp.Vx_.noalias() += Qxw * ff;
  vp.Vxx_ = qparam.Qxx;
  vp.Vxx_.noalias() += Qxw * fb;
  vp.Vxx_.diagonal().array() += xreg_;
  PROXDDP_NOMALLOC_END;
  return true;
}

template <typename Scalar>
Scalar SolverProxDDP<Scalar>::nonlinear_rollout_impl(const Problem &problem,
                                                     Workspace &workspace,
                                                     const Results &results,
                                                     const Scalar alpha) const {
  using ExplicitDynData = ExplicitDynamicsDataTpl<Scalar>;
  using DynamicsModel = DynamicsModelTpl<Scalar>;
  using DynamicsData = DynamicsDataTpl<Scalar>;

  const std::size_t nsteps = workspace.nsteps;
  std::vector<VectorXs> &xs = workspace.trial_xs;
  std::vector<VectorXs> &us = workspace.trial_us;
  std::vector<VectorXs> &lams = workspace.trial_lams;
  std::vector<VectorXs> &dyn_slacks = workspace.dyn_slacks;
  TrajOptData &prob_data = workspace.problem_data;

  {
    problem.init_state_error_->evaluate(xs[0], us[0], xs[1],
                                        prob_data.getInitData());
    compute_dir_x0(problem, workspace, results);
    const StageModel &stage = *problem.stages_[0];
    // use lams[0] as a tmp var for alpha * dx0
    if (!is_x0_fixed_) {
      lams[0] = alpha * workspace.dxs[0];
      stage.xspace().integrate(results.xs[0], lams[0], xs[0]);
      lams[0] = results.lams[0] + alpha * workspace.dlams[0];
    }
  }

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];
    StageData &data = prob_data.getStageData(i);

    const int nu = stage.nu();
    const int ndual = stage.numDual();

    ConstVectorRef ff = results.getFeedforward(i);
    ConstMatrixRef fb = results.getFeedback(i);
    auto ff_u = ff.head(nu);
    auto fb_u = fb.topRows(nu);
    auto ff_lm = ff.tail(ndual);
    auto fb_lm = fb.bottomRows(ndual);

    const VectorRef &dx = workspace.dxs[i];
    VectorRef &du = workspace.dus[i];
    du = alpha * ff_u;
    du.noalias() += fb_u * dx;
    stage.uspace().integrate(results.us[i], du, us[i]);

    VectorRef &dlam = workspace.dlams[i + 1];
    dlam = alpha * ff_lm;
    dlam.noalias() += fb_lm * dx;
    lams[i + 1] = results.lams[i + 1] + dlam;

    stage.evaluate(xs[i], us[i], xs[i + 1], data);

    // compute multiple-shooting gap
    CstrALWeightStrat weight_strat(mu_penal_, true);
    const ConstraintStack &cstr_stack = stage.constraints_;
    const ConstVectorRef dynlam =
        cstr_stack.getConstSegmentByConstraint(lams[i + 1], 0);
    const ConstVectorRef dynprevlam =
        cstr_stack.getConstSegmentByConstraint(workspace.lams_prev[i + 1], 0);
    dyn_slacks[i] = weight_strat.get(0) * (dynprevlam - dynlam);

    DynamicsData &dd = data.dyn_data();

    // lambda to be called in both branches
    auto explicit_model_update_xnext = [&]() {
      ExplicitDynData &exp_dd = static_cast<ExplicitDynData &>(dd);
      stage.xspace_next().integrate(exp_dd.xnext_, dyn_slacks[i], xs[i + 1]);
      // at xs[i+1], the dynamics gap = the slack dyn_slack[i].
      exp_dd.value_ = -dyn_slacks[i];
    };

    if (stage.has_dyn_model()) {
      const DynamicsModel &dm = stage.dyn_model();
      if (dm.is_explicit()) {
        explicit_model_update_xnext();
      } else {
        forwardDynamics(dm, xs[i], us[i], dd, xs[i + 1], 1, &dyn_slacks[i]);
      }
    } else {
      // otherwise assume explicit dynamics model
      explicit_model_update_xnext();
    }

    VectorRef &dx_next = workspace.dxs[i + 1];
    stage.xspace_next().difference(results.xs[i + 1], xs[i + 1], dx_next);

    PROXDDP_RAISE_IF_NAN_NAME(xs[i + 1], fmt::format("xs[{:d}]", i + 1));
    PROXDDP_RAISE_IF_NAN_NAME(us[i], fmt::format("us[{:d}]", i));
    PROXDDP_RAISE_IF_NAN_NAME(lams[i + 1], fmt::format("lams[{:d}]", i + 1));
  }

  // TERMINAL NODE
  problem.term_cost_->evaluate(xs[nsteps], problem.unone_,
                               *prob_data.term_cost_data);

  for (std::size_t k = 0; k < problem.term_cstrs_.size(); ++k) {
    const ConstraintType &tc = problem.term_cstrs_[k];
    FunctionData &td = *prob_data.term_cstr_data[k];
    tc.func->evaluate(xs[nsteps], problem.unone_, xs[nsteps], td);
  }

  // update multiplier
  if (!problem.term_cstrs_.empty()) {
    VectorRef &dlam = workspace.dlams.back();
    const VectorRef &dx = workspace.dxs.back();
    auto ff = results.getFeedforward(nsteps);
    auto fb = results.getFeedback(nsteps);
    dlam = alpha * ff;
    dlam.noalias() += fb * dx;
    lams.back() = results.lams.back() + dlam;
  }

  prob_data.cost_ = problem.computeTrajectoryCost(prob_data);
  return prob_data.cost_;
}

template <typename Scalar>
bool SolverProxDDP<Scalar>::run(const Problem &problem,
                                const std::vector<VectorXs> &xs_init,
                                const std::vector<VectorXs> &us_init,
                                const std::vector<VectorXs> &lams_init) {
  if (workspace_ == 0 || results_ == 0) {
    PROXDDP_RUNTIME_ERROR("workspace and results were not allocated yet!");
  }
  Workspace &workspace = *workspace_;
  Results &results = *results_;

  check_trajectory_and_assign(problem, xs_init, us_init, results.xs,
                              results.us);
  if (lams_init.size() == results.lams.size()) {
    for (std::size_t i = 0; i < lams_init.size(); i++) {
      long size = std::min(lams_init[i].rows(), results.lams[i].rows());
      results.lams[i].head(size) = lams_init[i].head(size);
    }
  }

  if (is_x0_fixed_) {
    workspace.trial_xs[0] = problem.getInitState();
  }

  logger.active = (verbose_ > 0);
  logger.start();

  set_penalty_mu(mu_init);
  set_rho(rho_init);
  xreg_ = reg_init;
  ureg_ = reg_init;

  workspace.prev_xs = results.xs;
  workspace.prev_us = results.us;
  workspace.lams_prev = results.lams;

  inner_tol_ = inner_tol0;
  prim_tol_ = prim_tol0;
  update_tols_on_failure();

  inner_tol_ = std::max(inner_tol_, target_tol_);
  prim_tol_ = std::max(prim_tol_, target_tol_);

  bool &conv = results.conv = false;

  results.al_iter = 0;
  results.num_iters = 0;
  std::size_t &al_iter = results.al_iter;
  while ((al_iter < max_al_iters) && (results.num_iters < max_iters)) {
    bool inner_conv = innerLoop(problem, workspace, results);
#ifndef NDEBUG
    {
      std::FILE *fi = std::fopen("pddp.log", "a");
      fmt::print(fi, "  p={:5.3e} | d={:5.3e}\n", results.prim_infeas,
                 results.dual_infeas);
      std::fclose(fi);
    }
#endif
    if (!inner_conv) {
      al_iter++;
      break;
    }

    // accept primal updates
    workspace.prev_xs = results.xs;
    workspace.prev_us = results.us;

    if (results.prim_infeas <= prim_tol_) {
      update_tols_on_success();

      switch (multiplier_update_mode) {
      case MultiplierUpdateMode::NEWTON:
        workspace.lams_prev = results.lams;
        break;
      case MultiplierUpdateMode::PRIMAL:
        workspace.lams_prev = workspace.lams_plus;
        break;
      case MultiplierUpdateMode::PRIMAL_DUAL:
        workspace.lams_prev = workspace.lams_pdal;
        break;
      default:
        break;
      }

      Scalar criterion = std::max(results.dual_infeas, results.prim_infeas);
      if (criterion <= target_tol_) {
        conv = true;
        break;
      }
    } else {
      Scalar old_mu = mu_penal_;
      bcl_update_alm_penalty();
      update_tols_on_failure();
      if (math::scalar_close(old_mu, mu_penal_)) {
        // reset penalty to initial value
        set_penalty_mu(mu_init);
      }
    }
    rho_penal_ *= bcl_params.rho_update_factor;

    inner_tol_ = std::max(inner_tol_, target_tol_);
    prim_tol_ = std::max(prim_tol_, target_tol_);

    al_iter++;
  }

  logger.finish(conv);
  return conv;
}

template <typename Scalar>
void SolverProxDDP<Scalar>::update_tols_on_failure() {
  prim_tol_ = prim_tol0 * std::pow(mu_penal_, bcl_params.prim_alpha);
  inner_tol_ = inner_tol0 * std::pow(mu_penal_, bcl_params.dual_alpha);
}

template <typename Scalar>
void SolverProxDDP<Scalar>::update_tols_on_success() {
  prim_tol_ = prim_tol_ * std::pow(mu_penal_, bcl_params.prim_beta);
  inner_tol_ = inner_tol_ * std::pow(mu_penal_, bcl_params.dual_beta);
}

template <typename Scalar>
Scalar
SolverProxDDP<Scalar>::forwardPass(const Problem &problem, Workspace &workspace,
                                   const Results &results, const Scalar alpha) {
  switch (rollout_type_) {
  case RolloutType::LINEAR:
    forward_linear_impl(problem, workspace, results, alpha);
    break;
  case RolloutType::NONLINEAR:
    nonlinear_rollout_impl(problem, workspace, results, alpha);
    break;
  default:
    assert(false && "unknown RolloutType!");
    break;
  }
  // computeProxTerms(workspace.trial_xs, workspace.trial_us, workspace);
  computeMultipliers(problem, workspace, workspace.trial_lams);
  return PDALFunction<Scalar>::evaluate(this, problem, workspace.trial_lams,
                                        workspace);
}

template <typename Scalar>
bool SolverProxDDP<Scalar>::innerLoop(const Problem &problem,
                                      Workspace &workspace, Results &results) {

  auto merit_eval_fun = [&](Scalar a0) {
    return this->forwardPass(problem, workspace, results, a0);
  };

  LogRecord iter_log;

  std::size_t &iter = results.num_iters;
  std::size_t inner_step = 0;
  results.traj_cost_ =
      problem.evaluate(results.xs, results.us, workspace.problem_data);
  computeMultipliers(problem, workspace, results.lams);
  results.merit_value_ =
      PDALFunction<Scalar>::evaluate(this, problem, results.lams, workspace);

  while (iter < max_iters) {
    // ASSUMPTION: last evaluation in previous iterate
    // was during linesearch, at the current candidate solution (x,u).
    /// TODO: make this smarter using e.g. some caching mechanism
    problem.computeDerivatives(results.xs, results.us, workspace.problem_data);
    projectJacobians(problem, workspace);
    // computeProxTerms(results.xs, results.us, workspace);
    // computeProxDerivatives(results.xs, results.us, workspace);
    const Scalar phi0 = results.merit_value_;

    while (true) {
      const bool success = backwardPass(problem, workspace, results);
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
    computeCriterion(problem, workspace, results);

    Scalar outer_crit = std::max(results.dual_infeas, results.prim_infeas);
    if (outer_crit <= target_tol_)
      return true;

    bool inner_conv = (workspace.inner_criterion <= inner_tol_);
    if (inner_conv && (inner_step > 0))
      return true;

    /// TODO: remove these expensive computations
    /// only use Q-function params etc
    linearRollout(problem, workspace, results);
    Scalar dphi0_analytical = PDALFunction<Scalar>::directionalDerivative(
        this, problem, results.lams, workspace);
    Scalar dphi0 = dphi0_analytical; // value used for LS & logging

    // check if we can early stop
    if (std::abs(dphi0) <= ls_params.dphi_thresh)
      return true;

    // otherwise continue linesearch
    Scalar alpha_opt = 1;
    Scalar phi_new = linesearch_.run(merit_eval_fun, phi0, dphi0, alpha_opt);

    // accept the step
    results.xs = workspace.trial_xs;
    results.us = workspace.trial_us;
    results.lams = workspace.trial_lams;
    results.traj_cost_ = workspace.problem_data.cost_;
    results.merit_value_ = phi_new;
    PROXDDP_RAISE_IF_NAN_NAME(alpha_opt, "alpha_opt");
    PROXDDP_RAISE_IF_NAN_NAME(results.merit_value_, "results.merit_value");
    PROXDDP_RAISE_IF_NAN_NAME(results.traj_cost_, "results.traj_cost");

    iter_log.iter = iter + 1;
    iter_log.al_iter = results.al_iter + 1;
    iter_log.xreg = xreg_;
    iter_log.inner_crit = workspace.inner_criterion;
    iter_log.prim_err = results.prim_infeas;
    iter_log.dual_err = results.dual_infeas;
    iter_log.step_size = alpha_opt;
    iter_log.dphi0 = dphi0;
    iter_log.merit = phi_new;
    iter_log.dM = phi_new - phi0;

    if (alpha_opt <= ls_params.alpha_min)
      this->increase_reg();

    invokeCallbacks(workspace, results);
    logger.log(iter_log);

    iter++;
    inner_step++;
  }
  return false;
}

template <typename Scalar>
void SolverProxDDP<Scalar>::computeInfeasibilities(const Problem &problem,
                                                   Workspace &workspace,
                                                   Results &results) const {
  // modifying quantities such as Qu, Qy... is allowed
  PROXDDP_NOMALLOC_BEGIN;
  const TrajOptData &prob_data = workspace.problem_data;
  const std::size_t nsteps = workspace.nsteps;

  // PRIMAL INFEASIBILITIES

  std::vector<VectorXs> &shifted_constraints = workspace.shifted_constraints;

  const FunctionData &init_data = prob_data.getInitData();
  workspace.stage_prim_infeas[0](0) = math::infty_norm(init_data.value_);

  using FuncDataVec = std::vector<shared_ptr<FunctionData>>;
  auto execute_on_stack =
      [](const ConstraintStack &stack, const VectorXs &shift_cvals,
         const VectorXs &lambda, VectorXs &stage_infeas,
         const FuncDataVec &constraint_data, CstrALWeightStrat &&weight_strat) {
        for (std::size_t k = 0; k < stack.size(); k++) {
          const CstrSet &set = stack.getConstraintSet(k);

          // compute and project displaced constraint
          auto scval_k = stack.getSegmentByConstraint(shift_cvals, k);
          auto lam_i = stack.getConstSegmentByConstraint(lambda, k);
          VectorXs &v = constraint_data[k]->value_;
          scval_k = v + weight_strat.get(k) * lam_i;
          set.projection(scval_k, scval_k); // apply projection
          stage_infeas((long)k) = math::infty_norm(v - scval_k);
        }
      };

  // compute infeasibility of all stage constraints
#pragma omp parallel for num_threads(problem.getNumThreads())
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];
    const StageData &stage_data = prob_data.getStageData(i);
    VectorXs &stage_infeas = workspace.stage_prim_infeas[i + 1];
    execute_on_stack(stage.constraints_, shifted_constraints[i + 1],
                     results.lams[i + 1], stage_infeas,
                     stage_data.constraint_data,
                     CstrALWeightStrat(mu_penal_, true));
  }

  // compute infeasibility of terminal constraints
  if (!problem.term_cstrs_.empty()) {
    execute_on_stack(problem.term_cstrs_, shifted_constraints.back(),
                     results.lams.back(), workspace.stage_prim_infeas.back(),
                     prob_data.term_cstr_data,
                     CstrALWeightStrat(mu_penal_, true));
  }

  results.prim_infeas = math::infty_norm(workspace.stage_prim_infeas);

  PROXDDP_NOMALLOC_END;
}

template <typename Scalar>
void SolverProxDDP<Scalar>::computeCriterion(const Problem &problem,
                                             Workspace &workspace,
                                             Results &results) const {
  // DUAL INFEASIBILITIES

  const std::size_t nsteps = workspace.nsteps;
  TrajOptData &prob_data = workspace.problem_data;

  {
    const int ndx = problem.init_state_error_->ndx1;
    VectorRef kkt_rhs = workspace.kkt_rhs_[0].col(0);
    auto kktx = kkt_rhs.head(ndx);
    if (is_x0_fixed_) {
      workspace.stage_inner_crits(0) = 0.;
      workspace.stage_dual_infeas(0) = 0.;
    } else {
      const ProxData &proxdata = *workspace.prox_datas[0];
      workspace.stage_inner_crits(0) = math::infty_norm(kkt_rhs);
      workspace.stage_dual_infeas(0) =
          math::infty_norm(kktx - rho() * proxdata.Lx_);
    }
  }

  for (std::size_t i = 1; i <= nsteps; i++) {
    const StageModel &st = *problem.stages_[i - 1];
    const int nu = st.nu();
    const int ndual = st.numDual();
    auto kkt_rhs = workspace.kkt_rhs_[i].col(0);
    auto kktu = kkt_rhs.head(nu);
    const auto kktlam = kkt_rhs.tail(ndual); // dual residual

    VParams &vp = workspace.value_params[i];

    Scalar rlam = math::infty_norm(kktlam);
    const ConstraintStack &cstr_mgr = st.constraints_;

    Scalar ru_ddp = 0.;
    const StageData &data = prob_data.getStageData(i - 1);
    const DynamicsDataTpl<Scalar> &dd = data.dyn_data();
    auto lam_head = cstr_mgr.getConstSegmentByConstraint(results.lams[i], 0);

    vp.Vx_ -= lam_head;
    kktu.noalias() += dd.Ju_.transpose() * vp.Vx_;
    vp.Vx_ += lam_head;
    ru_ddp = math::infty_norm(kktu);

#ifndef NDEBUG
    Scalar ru = math::infty_norm(kktu);
    auto gy = -lam_head + vp.Vx_;
    Scalar ry = math::infty_norm(gy);
    std::FILE *fi = std::fopen("pddp.log", "a");
    fmt::print(fi, "[{:>3d}]ru={:.2e},ry={:.2e},rlam={:.2e},ru_other={:.2e}\n",
               i, ru, ry, rlam, ru_ddp);
    std::fclose(fi);
#endif
    workspace.stage_inner_crits(long(i)) = std::max({ru_ddp, 0., rlam});
    {
      const CostData &proxdata = *workspace.prox_datas[i - 1];
      // const CostData &proxnext = *workspace.prox_datas[i];
      if (rho() > 0)
        kktu -= -rho() * proxdata.Lu_;
      Scalar dual_res_u = math::infty_norm(kktu);
      workspace.stage_dual_infeas(long(i)) = std::max(dual_res_u, 0.);
    }
  }
  workspace.inner_criterion = math::infty_norm(workspace.stage_inner_crits);
  results.dual_infeas = math::infty_norm(workspace.stage_dual_infeas);
}

} // namespace proxddp
