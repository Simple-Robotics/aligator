/// @file solver-proxddp.hxx
/// @brief  Implementations for the trajectory optimization algorithm.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/solver-proxddp.hpp"

#include <Eigen/Cholesky>

namespace proxddp {
template <typename Scalar>
void SolverProxDDP<Scalar>::computeDirection(const Problem &problem,
                                             Workspace &workspace,
                                             const Results &results) const {
  const std::size_t nsteps = problem.numSteps();

  // compute direction dx0
  {
    const value_store_t &vp = workspace.value_params[0];
    const StageModel &stage0 = problem.stages_[0];
    const FunctionData &init_data = *workspace.problem_data.init_data;
    const int ndual0 = problem.init_state_error.nr;
    const int ndx0 = stage0.ndx1();
    const VectorXs &lamin0 = results.lams_[0];
    const VectorXs &prevlam0 = workspace.prev_lams_[0];
    BlockXs kktmat0 = workspace.getKktView(ndx0, ndual0);
    Eigen::Block<BlockXs, -1, 1, true> kktrhs0 =
        workspace.getKktRhs(ndx0, ndual0, 1).col(0);
    kktmat0.setZero();
    kktrhs0.setZero();
    kktmat0.topLeftCorner(ndx0, ndx0) = vp.Vxx_;
    kktmat0.bottomLeftCorner(ndual0, ndx0) = init_data.Jx_;
    kktmat0.bottomRightCorner(ndual0, ndual0).diagonal().array() = -mu_;
    workspace.lams_plus_[0] = prevlam0 + mu_inverse_ * init_data.value_;
    workspace.lams_pdal_[0] = 2 * workspace.lams_plus_[0] - lamin0;
    kktrhs0.head(ndx0) = vp.Vx_ + init_data.Jx_ * lamin0;
    kktrhs0.tail(ndual0) = mu_ * (workspace.lams_plus_[0] - lamin0);

    auto kkt_sym = kktmat0.template selfadjointView<Eigen::Lower>();
    auto ldlt = kkt_sym.ldlt();
    workspace.pd_step_[0] = ldlt.solve(-kktrhs0);
  }

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = problem.stages_[i];
    VectorXs &pd_step = workspace.pd_step_[i + 1];
    Eigen::Block<const MatrixXs, -1, 1, true> feedforward =
        results.gains_[i].col(0);
    Eigen::Block<const MatrixXs, -1, -1, true> feedback =
        results.gains_[i].rightCols(stage.ndx1());

    pd_step = feedforward + feedback * workspace.dxs_[i];
  }
}

template <typename Scalar>
void SolverProxDDP<Scalar>::tryStep(const Problem &problem,
                                    Workspace &workspace,
                                    const Results &results,
                                    const Scalar alpha) const {

  const std::size_t nsteps = problem.numSteps();

  for (std::size_t i = 0; i <= nsteps; i++)
    workspace.trial_lams_[i] = results.lams_[i] + alpha * workspace.dlams_[i];

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = problem.stages_[i];
    stage.xspace_->integrate(results.xs_[i], alpha * workspace.dxs_[i],
                             workspace.trial_xs_[i]);
    stage.uspace_->integrate(results.us_[i], alpha * workspace.dus_[i],
                             workspace.trial_us_[i]);
  }
  const StageModel &stage = problem.stages_[nsteps - 1];
  stage.xspace_next_->integrate(results.xs_[nsteps],
                                alpha * workspace.dxs_[nsteps],
                                workspace.trial_xs_[nsteps]);
}

template <typename Scalar>
void SolverProxDDP<Scalar>::backwardPass(const Problem &problem,
                                         Workspace &workspace,
                                         Results &results) const {
  const TrajOptDataTpl<Scalar> &prob_data = workspace.problem_data;

  const std::size_t nsteps = problem.numSteps();

  /* Terminal node */
  const CostData &term_data = *prob_data.term_cost_data;
  value_store_t &term_value = workspace.value_params[nsteps];
  const CostData &proxdata = workspace.prox_datas[nsteps];

  term_value.v_2() = 2 * term_data.value_;
  term_value.Vx_ = term_data.Lx_ + rho_ * proxdata.Lx_;
  term_value.Vxx_ = term_data.Lxx_ + rho_ * proxdata.Lxx_;
  term_value.storage =
      term_value.storage.template selfadjointView<Eigen::Lower>();

  for (std::size_t i = 0; i < nsteps; i++) {
    computeGains(problem, workspace, results, nsteps - i - 1);
  }
  workspace.inner_criterion =
      math::infty_norm(workspace.inner_criterion_by_stage);
  results.dual_infeasibility = math::infty_norm(workspace.dual_infeas_by_stage);
}

template <typename Scalar>
void SolverProxDDP<Scalar>::computeGains(const Problem &problem,
                                         Workspace &workspace, Results &results,
                                         const std::size_t step) const {
  const StageModel &stage = problem.stages_[step];
  using ConstraintType = typename StageModel::Constraint;
  const std::size_t numc = stage.numConstraints();

  const value_store_t &vnext = workspace.value_params[step + 1];
  q_store_t &q_param = workspace.q_params[step];

  StageData &stage_data = *workspace.problem_data.stage_data[step];
  const CostData &cdata = *stage_data.cost_data;

  const CostData &proxdata = workspace.prox_datas[step];

  const int nprim = stage.numPrimal();
  const int ndual = stage.numDual();
  const int ndx1 = stage.ndx1();
  const int nu = stage.nu();
  const int ndx2 = stage.ndx2();

  assert(vnext.storage.rows() == ndx2 + 1);
  assert(vnext.storage.cols() == ndx2 + 1);

  // Use the contiguous full gradient/jacobian/hessian buffers
  // to fill in the Q-function derivatives
  q_param.storage.setZero();

  q_param.q_2() = 2 * cdata.value_;
  q_param.grad_.head(ndx1 + nu) = cdata.grad_ + rho_ * proxdata.grad_;
  q_param.grad_.tail(ndx2) = vnext.Vx_;
  q_param.hess_.topLeftCorner(ndx1 + nu, ndx1 + nu) =
      cdata.hess_ + rho_ * proxdata.hess_;
  q_param.hess_.bottomRightCorner(ndx2, ndx2) = vnext.Vxx_;

  // self-adjoint view to (nprim + ndual) sized block of kkt buffer
  BlockXs kkt_mat = workspace.getKktView(nprim, ndual);
  BlockXs kkt_rhs = workspace.getKktRhs(nprim, ndual, ndx1);
  Eigen::Block<BlockXs, -1, -1> kkt_jac = kkt_mat.block(nprim, 0, ndual, nprim);

  auto rhs_0 = kkt_rhs.col(0);
  auto rhs_D = kkt_rhs.rightCols(ndx1);

  const VectorXs &lam_inn = results.lams_[step + 1];
  const VectorXs &lamprev = workspace.prev_lams_[step + 1];
  VectorXs &lamplus = workspace.lams_plus_[step + 1];
  VectorXs &lampdal = workspace.lams_pdal_[step + 1];

  // Loop over constraints
  for (std::size_t j = 0; j < numc; j++) {
    const ConstraintType &cstr = stage.constraints_manager[j];
    FunctionData &cstr_data = *stage_data.constraint_data[j];
    MatrixXs &cstr_jac = cstr_data.jac_buffer_;

    // Grab Lagrange multiplier segments

    const auto laminn_j =
        stage.constraints_manager.getConstSegmentByConstraint(lam_inn, j);
    const auto lamprev_j =
        stage.constraints_manager.getConstSegmentByConstraint(lamprev, j);
    auto lamplus_j =
        stage.constraints_manager.getSegmentByConstraint(lamplus, j);
    auto lampdal_j =
        stage.constraints_manager.getSegmentByConstraint(lampdal, j);

    // compose Jacobian by projector and project multiplier
    const ConstraintSetBase<Scalar> &cstr_set = *cstr.set_;
    lamplus_j = lamprev_j + mu_inverse_ * cstr_data.value_;
    cstr_set.applyNormalConeProjectionJacobian(lamplus_j, cstr_jac);
    cstr_set.normalConeProjection(lamplus_j, lamplus_j);
    lampdal_j = 2 * lamplus_j - laminn_j;

    q_param.grad_.noalias() += cstr_jac.transpose() * laminn_j;
    q_param.hess_.noalias() += cstr_data.vhp_buffer_;

    // update the KKT jacobian columns
    stage.constraints_manager.getBlockByConstraint(kkt_jac, j) =
        cstr_jac.rightCols(nprim);
    stage.constraints_manager.getBlockByConstraint(rhs_D.bottomRows(ndual), j) =
        cstr_jac.leftCols(ndx1);
  }

  q_param.storage = q_param.storage.template selfadjointView<Eigen::Lower>();

  // blocks: u, y, and dual
  rhs_0.head(nprim) = q_param.grad_.tail(nprim);
  rhs_0.tail(ndual) =
      mu_ * (workspace.lams_plus_[step + 1] - results.lams_[step + 1]);

  rhs_D.middleRows(0, nu) = q_param.Qxu_.transpose();
  rhs_D.middleRows(nu, ndx2) = q_param.Qxy_.transpose();

  // KKT matrix: (u, y)-block = bottom right of q hessian
  kkt_mat.topLeftCorner(nprim, nprim) =
      q_param.hess_.bottomRightCorner(nprim, nprim);
  kkt_mat.bottomRightCorner(ndual, ndual).diagonal().array() = -mu_;

  workspace.inner_criterion_by_stage(long(step)) = math::infty_norm(rhs_0);
  workspace.dual_infeas_by_stage(long(step)) = math::infty_norm(
      rhs_0.head(nprim)); // dual infeas: norm of Q-function gradient

  /* Compute gains with LDLT */
  auto kkt_mat_view = kkt_mat.template selfadjointView<Eigen::Lower>();
  auto ldlt_ = kkt_mat_view.ldlt();
  results.gains_[step] = -kkt_rhs;
  ldlt_.solveInPlace(results.gains_[step]);

  /* Value function */
  value_store_t &v_current = workspace.value_params[step];
  v_current.storage = q_param.storage.topLeftCorner(ndx1 + 1, ndx1 + 1) +
                      kkt_rhs.transpose() * results.gains_[step];
}

template <typename Scalar>
void SolverProxDDP<Scalar>::solverInnerLoop(const Problem &problem,
                                            Workspace &workspace,
                                            Results &results) {
  // instantiate the subproblem merit function
  PDALFunction<Scalar> merit_fun{mu_, rho_, ls_params.mode};

  auto merit_eval_fun = [&](Scalar a0) {
    tryStep(problem, workspace, results, a0);
    problem.evaluate(workspace.trial_xs_, workspace.trial_us_,
                     workspace.trial_prob_data);
    this->evaluateProx(workspace.trial_xs_, workspace.trial_us_, workspace);
    return merit_fun.evaluate(problem, workspace.trial_lams_, workspace,
                              workspace.trial_prob_data);
  };

  std::size_t &k = results.num_iters;
  while (k < MAX_ITERS) {
    problem.evaluate(results.xs_, results.us_, workspace.problem_data);
    problem.computeDerivatives(results.xs_, results.us_,
                               workspace.problem_data);
    this->evaluateProx(results.xs_, results.us_, workspace);
    this->evaluateProxDerivatives(results.xs_, results.us_, workspace);

    backwardPass(problem, workspace, results);
    computeInfeasibilities(problem, workspace, results);

    if (verbose_ >= 1) {
      fmt::print(" | inner_crit: {:.3e}", workspace.inner_criterion);
      fmt::print(" | prim_err: {:.3e}\n", results.primal_infeasibility);
    }

    bool inner_conv = workspace.inner_criterion < inner_tol_;
    if (inner_conv) {
      break;
    } else {
      bool inner_acceptable = workspace.inner_criterion < target_tolerance;
      if (inner_acceptable) {
        if (results.primal_infeasibility < target_tolerance) {
          break;
        }
      }
    }

    if (verbose_ >= 1) {
      fmt::print(fmt::fg(fmt::color::yellow_green), "[iter {:>3d}]", k + 1);
      fmt::print("\n");
    }

    computeDirection(problem, workspace, results);

    Scalar phi0 = merit_fun.evaluate(problem, results.lams_, workspace,
                                     workspace.problem_data);
    Scalar eps = 1e-10;
    Scalar veps = merit_eval_fun(eps);
    Scalar dphi0 = (veps - phi0) / eps;

    Scalar alpha_opt = 1;

    switch (ls_params.strategy) {
    case LinesearchStrategy::ARMIJO:
      proxnlp::ArmijoLinesearch<Scalar>::run(
          merit_eval_fun, phi0, dphi0, verbose_, ls_params.ls_beta,
          ls_params.armijo_c1, ls_params.alpha_min, alpha_opt);
      break;
    case LinesearchStrategy::CUBIC_INTERP:
      proxnlp::CubicInterpLinesearch<Scalar>::run(
          merit_eval_fun, phi0, dphi0, verbose_, ls_params.armijo_c1,
          ls_params.alpha_min, alpha_opt);
      break;

    default:
      break;
    }

    results.traj_cost_ = merit_fun.traj_cost;
    results.merit_value_ = merit_fun.value_;
    if (verbose_ >= 1) {
      fmt::print(" | step size: {:.3e}, dphi0 = {:.3e}\n", alpha_opt, dphi0);
      fmt::print(" | merit value: {:.3e}\n", results.merit_value_);
    }

    // accept the step
    results.xs_ = workspace.trial_xs_;
    results.us_ = workspace.trial_us_;
    results.lams_ = workspace.trial_lams_;

    invokeCallbacks(workspace, results);

    k++;
  }
}

template <typename Scalar>
void SolverProxDDP<Scalar>::computeInfeasibilities(const Problem &problem,
                                                   Workspace &workspace,
                                                   Results &results) const {
  const TrajOptDataTpl<Scalar> &prob_data = workspace.problem_data;
  const std::size_t nsteps = problem.numSteps();
  results.primal_infeasibility = 0.;
  Scalar infeas_over_j = 0.;
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageData &sd = *prob_data.stage_data[i];
    const auto &cstr_mgr = problem.stages_[i].constraints_manager;
    infeas_over_j = 0.;
    for (std::size_t j = 0; j < cstr_mgr.numConstraints(); j++) {
      const ConstraintSetBase<Scalar> &cstr_set = *cstr_mgr[j].set_;
      auto &v = sd.constraint_data[j]->value_;
      /// @todo fix allocation here
      VectorXs vproj = v;
      cstr_set.normalConeProjection(v, vproj);
      infeas_over_j = std::max(infeas_over_j, math::infty_norm(vproj));
    }
    workspace.primal_infeas_by_stage(long(i)) = infeas_over_j;
  }
  results.primal_infeasibility =
      math::infty_norm(workspace.primal_infeas_by_stage);
  return;
}

} // namespace proxddp
