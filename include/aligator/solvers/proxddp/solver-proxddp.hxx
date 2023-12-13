/// @file solver-proxddp.hxx
/// @brief  Implementations for the trajectory optimization algorithm.
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "./solver-proxddp.hpp"
#include "aligator/core/iterative-refinement.hpp"
#include <boost/variant/apply_visitor.hpp>
#ifndef NDEBUG
#include <fmt/ostream.h>
#endif

namespace aligator {

template <typename Scalar>
SolverProxDDP<Scalar>::SolverProxDDP(const Scalar tol, const Scalar mu_init,
                                     const Scalar rho_init,
                                     const std::size_t max_iters,
                                     VerboseLevel verbose,
                                     HessianApprox hess_approx)
    : target_tol_(tol), mu_init(mu_init), rho_init(rho_init), verbose_(verbose),
      hess_approx_(hess_approx), ldlt_algo_choice_(LDLTChoice::DENSE),
      max_iters(max_iters), rollout_max_iters(1), linesearch_(ls_params) {
  ls_params.interp_type = proxsuite::nlp::LSInterpolation::CUBIC;
}

template <typename Scalar>
void SolverProxDDP<Scalar>::linearRollout(const Problem &problem) {
  ALIGATOR_NOMALLOC_BEGIN;
  compute_dir_x0(problem);

  const std::size_t nsteps = workspace_.nsteps;

  for (std::size_t i = 0; i < nsteps; i++) {
    VectorXs &pd_step = workspace_.pd_step_[i + 1];
    const auto ff = results_.getFeedforward(i);
    const auto fb = results_.getFeedback(i);

    pd_step = ff;
    pd_step.noalias() += fb * workspace_.dxs[i];
  }
  if (!problem.term_cstrs_.empty()) {
    const auto ff = results_.getFeedforward(nsteps);
    const auto fb = results_.getFeedback(nsteps);
    VectorRef &dlam = workspace_.dlams.back();
    dlam = ff;
    dlam.noalias() += fb * workspace_.dxs[nsteps];
  }
  ALIGATOR_NOMALLOC_END;
}

template <typename Scalar>
Scalar SolverProxDDP<Scalar>::forward_linear_impl(const Problem &problem,
                                                  Workspace &workspace,
                                                  const Results &results,
                                                  const Scalar alpha) {

  const std::size_t nsteps = workspace.nsteps;

  for (std::size_t i = 0; i < results.lams.size(); i++) {
    workspace.trial_lams[i] = results.lams[i] + alpha * workspace.dlams[i];
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
void SolverProxDDP<Scalar>::compute_dir_x0(const Problem &problem) {
  ALIGATOR_NOMALLOC_BEGIN;
  // compute direction dx0
  const VParams &vp = workspace_.value_params[0];
  const StageFunctionData &init_data = *workspace_.problem_data.init_data;
  const int ndual0 = problem.init_condition_->nr;
  const int ndx0 = problem.init_condition_->ndx1;
  const VectorXs &lampl0 = workspace_.lams_plus[0];
  const VectorXs &lamin0 = results_.lams[0];
  MatrixXs &kkt_mat = workspace_.kkt_mats_[0];
  VectorRef kkt_rhs = workspace_.kkt_rhs_[0].col(0);
  VectorRef kktx = kkt_rhs.head(ndx0);
  assert(kkt_rhs.size() == ndx0 + ndual0);
  assert(kkt_mat.cols() == ndx0 + ndual0);

  if (force_initial_condition_) {
    workspace_.pd_step_[0].setZero();
    workspace_.dxs[0] = -init_data.value_;
    workspace_.dlams[0] = -results_.lams[0];
    kkt_rhs.setZero();
  } else {
    auto kktl = kkt_rhs.tail(ndual0);
    kktx = vp.Vx_;
    kktx.noalias() += init_data.Jx_.transpose() * lamin0;
    kktl = mu() * (lampl0 - lamin0);

    auto kkt_xx = kkt_mat.topLeftCorner(ndx0, ndx0);
    kkt_xx = vp.Vxx_ + init_data.Hxx_;

    kkt_mat.topRightCorner(ndx0, ndual0) = init_data.Jx_.transpose();
    kkt_mat.bottomLeftCorner(ndual0, ndx0) = init_data.Jx_;
    kkt_mat.bottomRightCorner(ndual0, ndual0).diagonal().array() = -mu();
    auto &ldlt = workspace_.ldlts_[0];
    ALIGATOR_NOMALLOC_END;
    boost::apply_visitor([&](auto &&fac) { fac.compute(kkt_mat); }, ldlt);
    auto &resdl = workspace_.kkt_resdls_[0];
    auto &gains = workspace_.pd_step_[0];
    boost::apply_visitor(
        IterativeRefinementVisitor<Scalar>{kkt_mat, kkt_rhs, resdl, gains,
                                           refinement_threshold_,
                                           max_refinement_steps_},
        ldlt);
  }
  ALIGATOR_NOMALLOC_END;
}

template <typename Scalar>
void SolverProxDDP<Scalar>::setup(const Problem &problem) {
  workspace_ = Workspace(problem, ldlt_algo_choice_);
  results_ = Results(problem);
  linesearch_.setOptions(ls_params);

  workspace_.configureScalers(problem, mu_penal_,
                              applyDefaultScalingStrategy<Scalar>);
}

template <typename Scalar>
auto SolverProxDDP<Scalar>::backwardPass(const Problem &problem)
    -> BackwardRet {
  /* Terminal node */
  computeTerminalValue(problem);

  const std::size_t nsteps = workspace_.nsteps;
  for (std::size_t i = 0; i < nsteps; i++) {
    std::size_t t = nsteps - i - 1;
    updateHamiltonian(problem, t);
    assembleKktSystem(problem, t);
    BackwardRet b = computeGains(problem, t);
    if (b != BWD_SUCCESS) {
      return b;
    }
  }
  return BWD_SUCCESS;
}

template <typename Scalar>
void SolverProxDDP<Scalar>::computeMultipliers(
    const Problem &problem, const std::vector<VectorXs> &lams) {

  TrajOptData &prob_data = workspace_.problem_data;
  const std::size_t nsteps = workspace_.nsteps;

  std::vector<VectorXs> &lams_prev = workspace_.prev_lams;
  std::vector<VectorXs> &lams_plus = workspace_.lams_plus;
  std::vector<VectorXs> &lams_pdal = workspace_.lams_pdal;
  std::vector<VectorXs> &Lds = workspace_.Lds_;
  std::vector<VectorXs> &shifted_constraints = workspace_.shifted_constraints;

  // initial constraint
  {
    const VectorXs &lam0 = lams[0];
    const VectorXs &plam0 = lams_prev[0];
    StageFunctionData &data = *prob_data.init_data;
    shifted_constraints[0] = data.value_ + mu() * plam0;
    lams_plus[0] = shifted_constraints[0] * mu_inv();
    lams_pdal[0] = shifted_constraints[0] - 0.5 * mu() * lam0;
    lams_pdal[0] *= 2. * mu_inv();
    /// TODO: generalize to the other types of initial constraint (non-equality)
  }

  using FuncDataVec = std::vector<shared_ptr<StageFunctionData>>;
  auto execute_on_stack =
      [dual_weight = dual_weight](
          const ConstraintStack &stack, const VectorXs &lambda_,
          const VectorXs &prevlam_, VectorXs &lamplus_, VectorXs &lampdal_,
          VectorXs &lagrGd_, VectorXs &cvals_,
          typename Workspace::VecBool &active_,
          const FuncDataVec &constraint_data, CstrProximalScaler &scaler) {
        // k: constraint count variable
        using BlkView = BlkMatrix<ConstVectorRef, -1, 1>;
        using BlkViewMut = BlkMatrix<VectorRef, -1, 1>;
        using BoolBlkView =
            BlkMatrix<Eigen::Ref<typename Workspace::VecBool>, -1, 1>;
        auto &dims = stack.getDims();
        BlkView lambda(lambda_, dims);
        BlkView prevlam(prevlam_, dims);
        BlkViewMut lamplus(lamplus_, dims);
        BlkViewMut lampdal(lampdal_, dims);
        BlkViewMut lagrGd(lagrGd_, dims);
        BlkViewMut scval(cvals_, dims);
        BoolBlkView active(active_, dims);
        for (std::size_t k = 0; k < stack.size(); k++) {
          const CstrSet &set = *stack[k].set;
          const StageFunctionData &data = *constraint_data[k];

          Scalar m = scaler.get(k);
          scval[k] = data.value_ + m * prevlam[k];
          lampdal[k] = scval[k] - 0.5 * m * lambda[k];
          set.computeActiveSet(scval[k], active[k]);
          lampdal[k] = scval[k];

          set.normalConeProjection(scval[k], lamplus[k]);
          set.normalConeProjection(lampdal[k], lampdal[k]);

          // set multiplier = 1/mu * normal_proj(shifted_cstr)
          lamplus[k] /= m;
          lampdal[k] *= 2. / m;
          // compute prox Lagrangian dual gradient
          lagrGd[k] = m * (lamplus[k] - lambda[k]);
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

    execute_on_stack(cstr_stack, lami, plami, lamplusi, lampdali, Lds[i + 1],
                     shiftcvali, workspace_.active_constraints[i + 1],
                     sdata.constraint_data, workspace_.cstr_scalers[i]);
  }

  if (!problem.term_cstrs_.empty()) {
    execute_on_stack(problem.term_cstrs_, lams.back(), lams_prev.back(),
                     lams_plus.back(), lams_pdal.back(), Lds.back(),
                     shifted_constraints.back(),
                     workspace_.active_constraints.back(),
                     prob_data.term_cstr_data, workspace_.cstr_scalers.back());
  }
}

template <typename Scalar>
void SolverProxDDP<Scalar>::updateHamiltonian(const Problem &problem,
                                              const std::size_t t) {
  ALIGATOR_NOMALLOC_BEGIN;

  TrajOptData &pd = workspace_.problem_data;
  const StageModel &stage = *problem.stages_[t];
  const VParams &vnext = workspace_.value_params[t + 1];
  QParams &qparam = workspace_.q_params[t];

  StageData &stage_data = *pd.stage_data[t];
  const CostData &cdata = *stage_data.cost_data;

  qparam.q_ = cdata.value_;
  qparam.Qx = workspace_.Lxs_[t];
  qparam.Qu = workspace_.Lus_[t];

  int ndx1 = stage.ndx1();
  int nu = stage.nu();
  auto qpar_xu = qparam.hess_.topLeftCorner(ndx1 + nu, ndx1 + nu);
  qpar_xu = cdata.hess_;
  qparam.Qyy = vnext.Vxx_;
  qparam.Quu.diagonal().array() += ureg_;

  const ConstraintStack &cstr_stack = stage.constraints_;
  for (std::size_t k = 0; k < cstr_stack.size(); k++) {
    StageFunctionData &cstr_data = *stage_data.constraint_data[k];
    if (hess_approx_ == HessianApprox::EXACT) {
      qparam.hess_ += cstr_data.vhp_buffer_;
    }
  }
  ALIGATOR_NOMALLOC_END;
}

template <typename Scalar>
void SolverProxDDP<Scalar>::computeTerminalValue(const Problem &problem) {
  ALIGATOR_NOMALLOC_BEGIN;
  const std::size_t nsteps = workspace_.nsteps;

  const CostData &term_cost_data = *workspace_.problem_data.term_cost_data;
  const std::vector<shared_ptr<StageFunctionData>> &cstr_datas =
      workspace_.problem_data.term_cstr_data;

  VParams &term_value = workspace_.value_params[nsteps];
  term_value.v_ = term_cost_data.value_;
  term_value.Vx_ = workspace_.Lxs_[nsteps];
  term_value.Vxx_ = term_cost_data.Lxx_;
  term_value.Vxx_.diagonal().array() += xreg_;

  const ConstraintStack &cstr_mgr = problem.term_cstrs_;
  if (!cstr_mgr.empty()) {
    /* check number of multipliers */
    assert(results_.lams.size() == (nsteps + 2));
    assert(results_.gains_.size() == (nsteps + 1));
    const VectorXs &shift_cstr_v = workspace_.shifted_constraints[nsteps + 1];
    const VectorXs &prevlam = workspace_.prev_lams[nsteps + 1];
    const VectorXs &lamplus = workspace_.lams_plus[nsteps + 1];
    const VectorXs &lamin = results_.lams[nsteps + 1];
    auto ff = results_.getFeedforward(nsteps);
    auto fb = results_.getFeedback(nsteps);
    auto &dims = cstr_mgr.getDims();
    MatrixXs &pJx = workspace_.proj_jacobians.back();
    BlkMatrix<ConstVectorRef, -1, 1> scvv(shift_cstr_v, dims);
    BlkMatrix<MatrixRef, -1, 1> bkpJx(pJx, dims);
    BlkMatrix<VectorRef, -1, 1> bkff(ff, dims);
    BlkMatrix<MatrixRef, -1, 1> bkfb(fb, dims, {1});

    for (std::size_t k = 0; k < cstr_mgr.size(); ++k) {
      const CstrSet &cstr_set = *cstr_mgr[k].set;
      const StageFunctionData &cstr_data = *cstr_datas[k];

      auto pJx_k = bkpJx.blockRow(k);
      pJx_k = cstr_data.Jx_;
      assert(pJx_k.rows() == cstr_mgr[k].nr());
      assert(pJx_k.cols() == cstr_mgr[k].func->ndx1);
      cstr_set.applyNormalConeProjectionJacobian(scvv[k], pJx_k);

      auto ffk = bkff.blockSegment(k);
      auto fbk = bkfb.blockRow(k);

      ffk = lamplus - lamin;
      fbk = mu_inv() * pJx_k;

      term_value.Vxx_ += cstr_data.Hxx_;
    }
    term_value.v_ +=
        0.5 * mu_inv() * (lamplus.squaredNorm() - prevlam.squaredNorm());
    term_value.Vx_.noalias() += pJx.transpose() * ff;
    term_value.Vxx_.noalias() += pJx.transpose() * fb;
  }
  ALIGATOR_NOMALLOC_END;
}

template <typename Scalar>
void SolverProxDDP<Scalar>::assembleKktSystem(const Problem &problem,
                                              const std::size_t t) {
  ALIGATOR_NOMALLOC_BEGIN;
  using ColXpr = typename MatrixXs::ColXpr;
  using ColsBlockXpr = typename MatrixXs::ColsBlockXpr;
  const StageModel &stage = *problem.stages_[t];
  TrajOptData &pd = workspace_.problem_data;

  QParams &qparam = workspace_.q_params[t];
  const VParams &vnext = workspace_.value_params[t + 1];

  const StageData &stage_data = *pd.stage_data[t];
  const int nprim = stage.numPrimal();
  const int ndual = stage.numDual();
  const int ndx1 = stage.ndx1();
  const int nu = stage.nu();
  const int ndx2 = stage.ndx2();

  const VectorXs &laminnr = results_.lams[t + 1];
  const VectorXs &shift_cstr = workspace_.shifted_constraints[t + 1];
  const VectorXs &Ld = workspace_.Lds_[t + 1];

  MatrixXs &kkt_mat = workspace_.kkt_mats_[t + 1];
  MatrixXs &kkt_rhs = workspace_.kkt_rhs_[t + 1];

  assert(kkt_mat.rows() == (nprim + ndual));
  assert(kkt_rhs.rows() == (nprim + ndual));
  assert(kkt_rhs.cols() == (ndx1 + 1));

  // blocks of the KKT matrix
  auto kkt_jac = kkt_mat.bottomLeftCorner(ndual, nprim);
  auto kkt_prim = kkt_mat.topLeftCorner(nprim, nprim);
  auto kkt_dual = kkt_mat.bottomRightCorner(ndual, ndual);

  ColXpr kkt_rhs_ff(kkt_rhs.col(0));
  ColsBlockXpr kkt_rhs_fb(kkt_rhs.rightCols(ndx1));

  auto kkt_rhs_u = kkt_rhs_ff.head(nu);
  auto kkt_rhs_y = kkt_rhs_ff.segment(nu, ndx2);
  kkt_rhs_u = qparam.Qu;
  kkt_rhs_y = vnext.Vx_;

  MatrixRef kkt_rhs_ux = kkt_rhs_fb.topRows(nu);
  MatrixRef kkt_rhs_yx = kkt_rhs_fb.middleRows(nu, ndx2);
  MatrixRef kkt_rhs_lx = kkt_rhs_fb.bottomRows(ndual);
  kkt_rhs_ux.transpose() = qparam.Qxu;
  kkt_rhs_yx.transpose() = qparam.Qxy;

  // KKT matrix: (u, y)-block = bottom right of q hessian
  kkt_prim.topLeftCorner(nu, nu) = qparam.Quu;
  kkt_prim.bottomLeftCorner(ndx2, nu) = qparam.Quy.transpose();
  kkt_prim.topRightCorner(nu, ndx2) = qparam.Quy;
  kkt_prim.bottomRightCorner(ndx2, ndx2) = vnext.Vxx_;

  using BlkVecView = BlkMatrix<ConstVectorRef, -1, 1>;
  using BlkVecViewMut = BlkMatrix<VectorRef, -1, 1>;
  using BlkMatView = BlkMatrix<MatrixRef, -1, 1>;

  const ConstraintStack &cstr_mgr = stage.constraints_;
  auto &dims = cstr_mgr.getDims();
  // memory buffer for the projected Jacobian matrix
  MatrixXs &projectedJac = workspace_.proj_jacobians[t + 1];
  assert(cstr_mgr.totalDim() == ndual);
  const CstrProximalScaler &weight_strat = workspace_.cstr_scalers[t];
  kkt_dual.diagonal() = -weight_strat.diagMatrix();

  BlkVecView lamView(laminnr, dims);
  BlkVecView scvView(shift_cstr, dims);
  BlkMatView projJacView(projectedJac, dims);
  BlkMatView kktJacView(kkt_jac, dims);
  BlkVecView lagDualView(Ld, dims);
  BlkMatView kktRhsLxView(kkt_rhs_lx, dims);
  BlkVecViewMut kktRhsDual(kkt_rhs_ff.tail(ndual), dims);

  // Loop over constraints
  for (std::size_t j = 0; j < stage.numConstraints(); j++) {
    const CstrSet &cstr_set = *cstr_mgr[j].set;
    const StageFunctionData &cstr_data = *stage_data.constraint_data[j];

    // project constraint jacobian
    auto jac_proj_j = projJacView.blockRow(j);
    jac_proj_j = cstr_data.jac_buffer_;
    cstr_set.applyNormalConeProjectionJacobian(scvView[j], jac_proj_j);
    auto Jx_proj = jac_proj_j.leftCols(ndx1);
    auto Juy_proj = jac_proj_j.rightCols(nprim);

    kktRhsLxView.blockRow(j) = Jx_proj;
    kktJacView.blockRow(j) = Juy_proj;

    // get j-th rhs dual gradient
    kktRhsDual[j] = lagDualView[j];

    auto Jx_orig = cstr_data.jac_buffer_.leftCols(ndx1);
    auto Juy_orig = cstr_data.jac_buffer_.rightCols(nprim);
    // // add correction to kkt rhs ff
    auto kkt_rhs_prim = kkt_rhs_ff.head(nprim);
    kkt_rhs_prim.noalias() +=
        (Juy_orig - Juy_proj).transpose() * lagDualView[j];
    qparam.Qx.noalias() += (Jx_orig - Jx_proj).transpose() * lagDualView[j];
  }
  kkt_mat = kkt_mat.template selfadjointView<Eigen::Lower>();
  ALIGATOR_NOMALLOC_END;
}

template <typename Scalar>
auto SolverProxDDP<Scalar>::computeGains(const Problem &problem,
                                         const std::size_t t) -> BackwardRet {
  ALIGATOR_NOMALLOC_BEGIN;
  const StageModel &stage = *problem.stages_[t];
  const QParams &qparam = workspace_.q_params[t];
  const int ndx1 = stage.ndx1();
  const int ndual = stage.numDual();
  MatrixXs &kkt_mat = workspace_.kkt_mats_[t + 1];
  MatrixXs &kkt_rhs = workspace_.kkt_rhs_[t + 1];
  MatrixXs &resdl = workspace_.kkt_resdls_[t + 1];
  MatrixXs &gains = results_.gains_[t];

  auto &ldlt = workspace_.ldlts_[t + 1];
  ALIGATOR_NOMALLOC_END;
  boost::apply_visitor([&](auto &&fac) { fac.compute(kkt_mat); }, ldlt);
  ALIGATOR_NOMALLOC_BEGIN;

  // check inertia
  {
    Eigen::VectorXi signature;
    boost::apply_visitor(proxsuite::nlp::ComputeSignatureVisitor{signature},
                         ldlt);
    // (n+, n-, n0)
    std::array<int, 3> inertia = proxsuite::nlp::computeInertiaTuple(signature);
    if ((inertia[2] > 0) || (inertia[1] != ndual)) {
      return BWD_WRONG_INERTIA;
    }
  }

  ALIGATOR_NOMALLOC_END;
  boost::apply_visitor(
      IterativeRefinementVisitor<Scalar>{kkt_mat, kkt_rhs, resdl, gains,
                                         refinement_threshold_,
                                         max_refinement_steps_},
      ldlt);
  ALIGATOR_NOMALLOC_BEGIN;

  /// Value function/Riccati update:
  /// provided by the Schur complement.

  VParams &vp = workspace_.value_params[t];
  auto kkt_rhs_fb = kkt_rhs.rightCols(ndx1);
  auto Qxw = kkt_rhs_fb.transpose();
  auto ff = results_.getFeedforward(t);
  auto fb = results_.getFeedback(t);

  vp.Vx_ = qparam.Qx;
  vp.Vx_.noalias() += Qxw * ff;
  vp.Vxx_ = qparam.Qxx;
  vp.Vxx_.noalias() += Qxw * fb;
  vp.Vxx_.diagonal().array() += xreg_;
  ALIGATOR_NOMALLOC_END;
  return BWD_SUCCESS;
}

template <typename Scalar>
Scalar SolverProxDDP<Scalar>::nonlinear_rollout_impl(const Problem &problem,
                                                     const Scalar alpha) {
  using ExplicitDynData = ExplicitDynamicsDataTpl<Scalar>;

  const std::size_t nsteps = workspace_.nsteps;
  std::vector<VectorXs> &xs = workspace_.trial_xs;
  std::vector<VectorXs> &us = workspace_.trial_us;
  std::vector<VectorXs> &lams = workspace_.trial_lams;
  std::vector<VectorRef> &dxs = workspace_.dxs;
  std::vector<VectorRef> &dus = workspace_.dus;
  const std::vector<VectorXs> &lams_prev = workspace_.prev_lams;
  std::vector<VectorXs> &dyn_slacks = workspace_.dyn_slacks;
  TrajOptData &prob_data = workspace_.problem_data;

  {
    compute_dir_x0(problem);
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

    VectorRef &dlam = workspace_.dlams[t + 1];
    dlam = alpha * ff_lm;
    dlam.noalias() += fb_lm * dxs[t];
    lams[t + 1] = results_.lams[t + 1] + dlam;

    stage.evaluate(xs[t], us[t], xs[t + 1], data);

    // compute desired multiple-shooting gap from the multipliers
    {
      const ConstraintStack &cstr_stack = stage.constraints_;
      using BlkView = BlkMatrix<ConstVectorRef, -1, 1>;
      const BlkView lamview(lams[t + 1], cstr_stack.getDims());
      const BlkView plamview(lams_prev[t + 1], cstr_stack.getDims());
      const auto &weight_strat = workspace_.cstr_scalers[t];
      dyn_slacks[t] = weight_strat.get(0) * (lamview[0] - plamview[0]);
    }

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
      ConstVectorRef slack = dyn_slacks[t];
      forwardDynamics<Scalar>::run(stage.dyn_model(), xs[t], us[t], dd,
                                   xs[t + 1], slack, rollout_max_iters);
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
    VectorRef &dlam = workspace_.dlams.back();
    const VectorRef &dx = workspace_.dxs.back();
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
bool SolverProxDDP<Scalar>::run(const Problem &problem,
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
  }

  logger.active = (verbose_ > 0);
  logger.printHeadline();

  set_penalty_mu(mu_init);
  set_rho(rho_init);

  workspace_.prev_xs = results_.xs;
  workspace_.prev_us = results_.us;
  workspace_.prev_lams = results_.lams;

  inner_tol_ = inner_tol0;
  prim_tol_ = prim_tol0;
  update_tols_on_failure();

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
      update_tols_on_success();

      switch (multiplier_update_mode) {
      case MultiplierUpdateMode::NEWTON:
        workspace_.prev_lams = results_.lams;
        break;
      case MultiplierUpdateMode::PRIMAL:
        workspace_.prev_lams = workspace_.lams_plus;
        break;
      case MultiplierUpdateMode::PRIMAL_DUAL:
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
      bcl_update_alm_penalty();
      update_tols_on_failure();
      if (math::scalar_close(old_mu, mu_penal_)) {
        // reset penalty to initial value
        set_penalty_mu(mu_init);
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
Scalar SolverProxDDP<Scalar>::forwardPass(const Problem &problem,
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
  computeMultipliers(problem, workspace_.trial_lams);
  return PDALFunction<Scalar>::evaluate(*this, problem, workspace_.trial_lams,
                                        workspace_);
}

template <typename Scalar>
bool SolverProxDDP<Scalar>::innerLoop(const Problem &problem) {

  auto merit_eval_fun = [&](Scalar a0) -> Scalar {
    return forwardPass(problem, a0);
  };

  LogRecord iter_log;

  std::size_t &iter = results_.num_iters;
  std::size_t inner_step = 0;
  results_.traj_cost_ =
      problem.evaluate(results_.xs, results_.us, workspace_.problem_data);
  computeMultipliers(problem, results_.lams);
  results_.merit_value_ =
      PDALFunction<Scalar>::evaluate(*this, problem, results_.lams, workspace_);

  for (; iter < max_iters; iter++) {
    // ASSUMPTION: last evaluation in previous iterate
    // was during linesearch, at the current candidate solution (x,u).
    /// TODO: make this smarter using e.g. some caching mechanism
    problem.computeDerivatives(results_.xs, results_.us,
                               workspace_.problem_data);
    const Scalar phi0 = results_.merit_value_;

    LagrangianDerivatives<Scalar>::compute(problem, workspace_.problem_data,
                                           results_.lams, workspace_.Lxs_,
                                           workspace_.Lus_);
    if (force_initial_condition_) {
      workspace_.Lxs_[0].setZero();
    }
    computeInfeasibilities(problem);
    computeCriterion(problem);

    Scalar outer_crit = std::max(results_.dual_infeas, results_.prim_infeas);
    if (outer_crit <= target_tol_)
      return true;

    // attempt backward pass until successful
    // i.e. no inertia problems
    initialize_regularization();
    while (true) {
      BackwardRet b = backwardPass(problem);
      switch (b) {
      case BWD_SUCCESS:
        break;
      case BWD_WRONG_INERTIA: {
        if (xreg_ >= reg_max)
          return false;
        increase_regularization();
        continue;
      }
      }
      break; // if you broke from the switch
    }

    bool inner_conv = (workspace_.inner_criterion <= inner_tol_);
    if (inner_conv && (inner_step > 0))
      return true;

    /// TODO: remove these expensive computations
    /// only use Q-function params etc
    linearRollout(problem);
    Scalar dphi0 = PDALFunction<Scalar>::directionalDerivative(
        *this, problem, results_.lams, workspace_);

    // check if we can early stop
    if (std::abs(dphi0) <= ls_params.dphi_thresh)
      return true;

    // otherwise continue linesearch
    Scalar alpha_opt = 1;
    Scalar phi_new = linesearch_.run(merit_eval_fun, phi0, dphi0, alpha_opt);

    // accept the step
    results_.xs = workspace_.trial_xs;
    results_.us = workspace_.trial_us;
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
      increase_regularization();
    }
    invokeCallbacks(workspace_, results_);
    logger.log(iter_log);

    xreg_last_ = xreg_;
    inner_step++;
  }
  return false;
}

template <typename Scalar>
void SolverProxDDP<Scalar>::computeInfeasibilities(const Problem &problem) {
  // modifying quantities such as Qu, Qy... is allowed
  ALIGATOR_NOMALLOC_BEGIN;
  const TrajOptData &prob_data = workspace_.problem_data;
  const std::size_t nsteps = workspace_.nsteps;

  // PRIMAL INFEASIBILITIES

  std::vector<VectorXs> &lams_plus = workspace_.lams_plus;
  std::vector<VectorXs> &lams_prev = workspace_.prev_lams;

  const StageFunctionData &init_data = *prob_data.init_data;
  workspace_.stage_prim_infeas[0](0) = math::infty_norm(init_data.value_);

  auto execute_on_stack = [](const ConstraintStack &stack,
                             const VectorXs &lams_plus,
                             const VectorXs &prev_lams, VectorXs &stage_infeas,
                             CstrProximalScaler &scaler) {
    auto e = scaler.apply(prev_lams - lams_plus);
    for (std::size_t j = 0; j < stack.size(); j++) {
      /// TODO Find an alternative to this
      stage_infeas((long)j) =
          math::infty_norm(stack.constSegmentByConstraint(e, j));
    }
  };

  // compute infeasibility of all stage constraints
#pragma omp parallel for num_threads(problem.getNumThreads())
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];
    VectorXs &stage_infeas = workspace_.stage_prim_infeas[i + 1];
    execute_on_stack(stage.constraints_, lams_plus[i + 1], lams_prev[i + 1],
                     stage_infeas, workspace_.cstr_scalers[i]);
  }

  // compute infeasibility of terminal constraints
  if (!problem.term_cstrs_.empty()) {
    execute_on_stack(problem.term_cstrs_, lams_plus.back(), lams_prev.back(),
                     workspace_.stage_prim_infeas.back(),
                     workspace_.cstr_scalers.back());
  }

  results_.prim_infeas = math::infty_norm(workspace_.stage_prim_infeas);

  ALIGATOR_NOMALLOC_END;
}

template <typename Scalar>
void SolverProxDDP<Scalar>::computeCriterion(const Problem &problem) {
  // DUAL INFEASIBILITIES
  const std::size_t nsteps = workspace_.nsteps;

  workspace_.stage_inner_crits.setZero();
  workspace_.stage_dual_infeas.setZero();
  Scalar x_residuals = 0.;
  Scalar u_residuals = 0.;
  if (!force_initial_condition_) {
    const int ndual = problem.stages_[0]->numDual();
    Scalar rx = math::infty_norm(workspace_.Lxs_[0]);
    VectorRef kkt_rhs = workspace_.kkt_mats_[0].col(0);
    auto kktlam = kkt_rhs.tail(ndual);
    Scalar rlam = math::infty_norm(kktlam);
    x_residuals = std::max(x_residuals, rx);
    workspace_.stage_inner_crits(0) = std::max(rx, rlam);
    workspace_.stage_dual_infeas(0) = rx;
  }

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &st = *problem.stages_[i];
    const int ndual = st.numDual();
    ConstVectorRef kkt_rhs = workspace_.kkt_rhs_[i + 1].col(0);
    ConstVectorRef kktlam = kkt_rhs.tail(ndual); // dual residual

    Scalar rlam = math::infty_norm(kktlam);
    Scalar rx = math::infty_norm(workspace_.Lxs_[i + 1]);
    Scalar ru = math::infty_norm(workspace_.Lus_[i]);
    x_residuals = std::max(x_residuals, rx);
    u_residuals = std::max(u_residuals, ru);

    rx *= 1e-3;
    workspace_.stage_inner_crits(long(i + 1)) = std::max({rx, ru, rlam});
    workspace_.stage_dual_infeas(long(i + 1)) = std::max(rx, ru);
  }

  workspace_.inner_criterion = math::infty_norm(workspace_.stage_inner_crits);
  results_.dual_infeas = math::infty_norm(workspace_.stage_dual_infeas);
}

} // namespace aligator
