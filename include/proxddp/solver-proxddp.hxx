/// @file solver-proxddp.hxx
/// @brief  Implementations for the trajectory optimization algorithm.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/solver-proxddp.hpp"


namespace proxddp
{
  template<typename Scalar>
  void SolverProxDDP<Scalar>::computeDirection(
    const Problem& problem,
    WorkspaceTpl<Scalar>& workspace) const
  {
    const std::size_t nsteps = problem.numSteps();

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
  void SolverProxDDP<Scalar>::computeActiveSetsAndMultipliers(
    const Problem& problem,
    Workspace& workspace,
    Results& results) const
  {

  }
  
  template<typename Scalar>
  void SolverProxDDP<Scalar>::tryStep(
    const Problem& problem,
    Workspace& workspace,
    const Results& results,
    const Scalar alpha) const
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
  void SolverProxDDP<Scalar>::backwardPass(
    const Problem& problem,
    Workspace& workspace,
    ResultsTpl<Scalar>& results) const
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
  void SolverProxDDP<Scalar>::computeGains(
    const ShootingProblemTpl<Scalar>& problem,
    Workspace& workspace,
    ResultsTpl<Scalar>& results,
    const std::size_t step) const
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

    workspace.inner_criterion_by_stage((long)step) = math::infty_norm(rhs_0);

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

} // namespace proxddp

