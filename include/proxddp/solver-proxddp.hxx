/// @file solver-proxddp.hxx
/// @brief  Implementations for the trajectory optimization algorithm.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/solver-proxddp.hpp"


namespace proxddp
{
  template<typename Scalar>
  void SolverProxDDP<Scalar>::
  computeDirection(const Problem& problem, Workspace& workspace) const
  {
    const std::size_t nsteps = problem.numSteps();

    // compute direction dx0
    {
      const auto& vp = workspace.value_params[0];
      const StageModel& stage0 = problem.stages_[0];
      const ShootingProblemDataTpl<Scalar> prob_data = *workspace.problem_data;
      const int ndual0 = problem.init_state_error.nr;
      const int ntot0 = stage0.ndx1() + ndual0;
      MatrixXs kktmat0(ntot0, ntot0);
      VectorXs kktrhs0(ntot0);
      kktmat0.setIdentity();
      kktrhs0.setZero();
      kktmat0.topLeftCorner(stage0.ndx1(), stage0.ndx1()) = vp.Vxx_;
      // kktmat0.bottomRightCorner(ndual0, ndual0) *= -mu_;
      kktrhs0.head(stage0.ndx1()) = vp.Vx_;
      kktrhs0.tail(ndual0) = prob_data.init_data->value_;

      auto ldlt = kktmat0.ldlt();
      ldlt.compute(kktmat0);
      auto res = ldlt.solve(-kktrhs0);
      workspace.dxs_[0] = res.head(stage0.ndx1());
    }

    workspace.dxs_[0].setZero();
    for (std::size_t i = 0; i < nsteps; i++)
    {
      const StageModel& stage = problem.stages_[i];
      // int nu = stage.nu();
      // int ndual = stage.numDual();
      VectorRef pd_step = workspace.pd_step_[i + 1];
      MatrixRef gains = workspace.gains_[i];
      ConstVectorRef feedforward = gains.col(0);
      ConstMatrixRef feedback = gains.rightCols(stage.ndx1());

      pd_step = feedforward + feedback * workspace.dxs_[i];
    }

  }

  template<typename Scalar>
  void SolverProxDDP<Scalar>::
  computeActiveSetsAndMultipliers(const Problem& problem, Workspace& workspace, Results& results) const
  {

  }
  
  template<typename Scalar>
  void SolverProxDDP<Scalar>::
  tryStep(const Problem& problem, Workspace& workspace, const Results& results, const Scalar alpha) const
  {

    const std::size_t nsteps = problem.numSteps();

    for (std::size_t i = 0; i < nsteps; i++)
    {
      const StageModel& stage = problem.stages_[i];
      stage.xspace1_.integrate(results.xs_[i], alpha * workspace.dxs_[i], workspace.trial_xs_[i]);
      stage.uspace_.integrate(results.us_[i], alpha * workspace.dus_[i], workspace.trial_us_[i]);
      workspace.trial_lams_[i] = results.lams_[i] + alpha * workspace.dlams_[i];
      if (i == nsteps - 1)
      {
        stage.xspace2_.integrate(results.xs_[nsteps], alpha * workspace.dxs_[nsteps], workspace.trial_xs_[nsteps]);
      }
    }
  }

  template<typename Scalar>
  void SolverProxDDP<Scalar>::
  backwardPass(const Problem& problem, Workspace& workspace, Results& results) const
  {
    const ShootingProblemDataTpl<Scalar>& problem_data = *workspace.problem_data;

    const std::size_t nsteps = problem.numSteps();

    /* Terminal node */
    const auto& term_data = *problem_data.term_cost_data;
    value_store_t& term_value = workspace.value_params[nsteps];

    term_value.v_2_ = 2 * term_data.value_;
    term_value.Vx_ = term_data.Lx_;
    term_value.Vxx_ = term_data.Lxx_;

    for (std::size_t i = 0; i < nsteps; i++)
    {
      computeGains(problem, workspace, results, nsteps - i - 1);
    }
    workspace.inner_criterion = math::infty_norm(workspace.inner_criterion_by_stage);
  }

  template<typename Scalar>
  void SolverProxDDP<Scalar>::
  computeGains(const Problem& problem, Workspace& workspace, Results& results, const std::size_t step) const
  {
    using FunctionData = FunctionDataTpl<Scalar>;

    const StageModel& stage = problem.stages_[step];
    const std::size_t numc = stage.numConstraints();

    const value_store_t& vnext = workspace.value_params[step + 1];
    q_store_t& q_param = workspace.q_params[step];

    StageDataTpl<Scalar>& stage_data = *workspace.problem_data->stage_data[step];
    const CostDataAbstract<Scalar>& cdata = *stage_data.cost_data;

    int nprim = stage.numPrimal();
    int ndual = stage.numDual();
    int ndx1 = stage.ndx1();
    int nu = stage.nu();
    int ndx2 = stage.ndx2();

    assert(vnext.storage.rows() == ndx2 + 1);
    assert(vnext.storage.cols() == ndx2 + 1);

    // Use the contiguous full gradient/jacobian/hessian buffers
    // to fill in the Q-function derivatives
    q_param.storage.setZero();

    q_param.q_2_ = 2 * cdata.value_;
    q_param.grad_.head(ndx1 + nu) = cdata.grad_;
    q_param.grad_.tail(ndx2) = vnext.Vx_;
    q_param.hess_.topLeftCorner(ndx1 + nu, ndx1 + nu) = cdata.hess_;
    q_param.hess_.bottomRightCorner(ndx2, ndx2) = vnext.Vxx_;

    // self-adjoint view to (nprim + ndual) sized block of kkt buffer
    MatrixRef kkt_mat = workspace.getKktView(nprim, ndual);
    MatrixRef kkt_rhs = workspace.getKktRhs(nprim, ndual, ndx1);
    MatrixRef kkt_jac = kkt_mat.block(nprim, 0, ndual, nprim);

    VectorRef rhs_0 = kkt_rhs.col(0);
    MatrixRef rhs_D = kkt_rhs.rightCols(ndx1);

    VectorRef lambda_in = results.lams_[step];
    VectorRef lamprev = workspace.prev_lams_[step];
    VectorRef lamplus = workspace.lams_plus_[step];
    VectorRef lampdal = workspace.lams_pdal_[step];

    // Loop over constraints
    for (std::size_t i = 0; i < numc; i++)
    {
      const auto& cstr = stage.constraints_manager[i];
      const auto& cstr_set = *cstr->set_;
      FunctionData& cstr_data = *stage_data.constraint_data[i];
      MatrixRef cstr_jac = cstr_data.jac_buffer_;

      // Grab Lagrange multiplier segments

      VectorRef lam_i     = stage.constraints_manager.getSegmentByConstraint(lambda_in, i);
      VectorRef lamprev_i = stage.constraints_manager.getSegmentByConstraint(lamprev,   i);
      VectorRef lamplus_i = stage.constraints_manager.getSegmentByConstraint(lamplus,   i);
      VectorRef lampdal_i = stage.constraints_manager.getSegmentByConstraint(lampdal,   i);

      assert(cstr_jac.rows() == cstr->nr());
      assert(cstr_jac.cols() == ndx1 + nprim);

      // compose Jacobian by projector and project multiplier
      lamplus_i = lamprev_i + mu_inverse_ * cstr_data.value_;
      cstr_set.applyNormalConeProjectionJacobian(lamplus_i, cstr_jac);
      lamplus_i.noalias() = cstr_set.normalConeProjection(lamplus_i);
      lampdal_i = 2 * lamplus_i - lam_i;

      q_param.grad_.noalias() += cstr_jac.transpose() * lam_i;
      q_param.hess_.noalias() += cstr_data.vhp_buffer_;

      // update the KKT jacobian columns
      stage.constraints_manager.getBlockByConstraint(kkt_jac, i) = cstr_jac.rightCols(nprim);
      stage.constraints_manager.getBlockByConstraint(rhs_D.bottomRows(ndual), i) = cstr_jac.leftCols(ndx1);
    }

    // blocks: u, y, and dual
    rhs_0.head(nprim) = q_param.grad_.tail(nprim);
    rhs_0.tail(ndual) = mu_ * (workspace.lams_plus_[step] - results.lams_[step]);

    rhs_D.middleRows(0,  nu) = q_param.Qxu_.transpose();
    rhs_D.middleRows(nu, ndx2) = q_param.Qxy_.transpose();

    // KKT matrix: (u, y)-block = bottom right of q hessian
    kkt_mat.topLeftCorner(nprim, nprim) = q_param.hess_.bottomRightCorner(nprim, nprim);
    kkt_mat.bottomRightCorner(ndual, ndual).diagonal().array() = -mu_;

    auto kkt_mat_view = kkt_mat.template selfadjointView<Eigen::Lower>();

    workspace.inner_criterion_by_stage(long(step)) = math::infty_norm(rhs_0);

    /* Compute gains with LDLT */
    auto ldlt_ = kkt_mat_view.ldlt();
    workspace.gains_[step] = -kkt_rhs;
    ldlt_.solveInPlace(workspace.gains_[step]);

    /* Value function */
    value_store_t& v_current = workspace.value_params[step];
    v_current.storage = \
      q_param.storage.topLeftCorner(ndx1 + 1, ndx1 + 1) +\
      kkt_rhs.transpose() * workspace.gains_[step];
  }

  template<typename Scalar>
  void SolverProxDDP<Scalar>::solverInnerLoop(const Problem& problem, Workspace& workspace, Results& results)
  {
    const std::size_t nsteps = problem.numSteps();

    assert(results.xs_.size() == nsteps + 1);
    assert(results.us_.size() == nsteps);
    assert(results.lams_.size() == nsteps);

    // instantiate the subproblem merit function
    PDAL_Function<Scalar> fun { mu_, ls_params.ls_mode };

    auto merit_eval_fun = [&](Scalar a0) {
      tryStep(problem, workspace, results, a0);
      return fun.evaluate(
        problem,
        workspace.trial_xs_,
        workspace.trial_us_,
        workspace.trial_lams_,
        workspace,
        *workspace.problem_data);
    };

    std::size_t& k = results.num_iters;
    while (k < MAX_STEPS)
    {
      problem.evaluate(results.xs_, results.us_, *workspace.problem_data);
      problem.computeDerivatives(results.xs_, results.us_, *workspace.problem_data);

      backwardPass(problem, workspace, results);

      fmt::print(" | inner_crit: {:.3e}\n", workspace.inner_criterion);

      if ((workspace.inner_criterion < inner_tol_))
      {
        break;
      }

      fmt::print(fmt::fg(fmt::color::yellow_green), "[iter {:>3d}]", k);
      fmt::print("\n");

      computeDirection(problem, workspace);

      Scalar phi0 = fun.evaluate(
        problem,
        results.xs_,
        results.us_,
        results.lams_,
        workspace,
        *workspace.problem_data);
      Scalar eps = 1e-8;
      Scalar veps = merit_eval_fun(eps);
      Scalar dphi0 = (veps - phi0) / eps;

      Scalar alpha_opt = 1;

      proxnlp::ArmijoLinesearch<Scalar>::run(
        merit_eval_fun, phi0, dphi0,
        ls_params.ls_beta, ls_params.armijo_c1, ls_params.alpha_min,
        alpha_opt);

      results.traj_cost_ = fun.traj_cost;
      fmt::print(" | step size: {:.3e}, dphi0 = {:.3e}\n", alpha_opt, dphi0);
      fmt::print(" | new merit fun. val: {:.3e}\n", phi0);

      // accept the damn step
      results.xs_ = workspace.trial_xs_;
      results.us_ = workspace.trial_us_;
      results.lams_ = workspace.trial_lams_;

      k++;
    }
  }

  template<typename Scalar>
  void SolverProxDDP<Scalar>::computeInfeasibilities(const Problem& problem, Workspace& workspace) const
  {
    const ShootingProblemDataTpl<Scalar>& prob_data = *workspace.problem_data;
    const std::size_t nsteps = problem.numSteps();
    auto& prim_infeases = workspace.primal_infeas_by_stage;
    workspace.primal_infeasibility = 0.;
    for (std::size_t i = 0; i < nsteps; i++)
    {
      const StageDataTpl<Scalar>& sd = *prob_data.stage_data[i];
      const auto& cstr_mgr = problem.stages_[i].constraints_manager;
      std::vector<Scalar> infeas_by_cstr(cstr_mgr.numConstraints());
      for (std::size_t j = 0; j < cstr_mgr.numConstraints(); j++)
      {
        const ConstraintSetBase<Scalar>& cstr_set = cstr_mgr[j]->getConstraintSet();
        infeas_by_cstr[j] = math::infty_norm(cstr_set.normalConeProjection(sd.constraint_data[j]->value_));
      }
      prim_infeases(long(i)) = *std::max_element(infeas_by_cstr.begin(), infeas_by_cstr.end());
    }
    workspace.primal_infeasibility = math::infty_norm(prim_infeases);
    return;
  }

} // namespace proxddp

