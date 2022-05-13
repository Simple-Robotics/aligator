/// @file solver-proxddp.hxx
/// @brief  Implementations for the trajectory optimization algorithm.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/solver-proxddp.hpp"


namespace proxddp
{
  template<typename Scalar>
  void SolverTpl<Scalar>::forwardPass(
    const ShootingProblemTpl<Scalar>& problem,
    WorkspaceTpl<Scalar>& workspace) const
  {
    using StageModel = StageModelTpl<Scalar>;

    const std::vector<MatrixXs>& gains = workspace.gains_;
    const std::size_t nsteps = problem.numSteps();
    assert(gains.size() == nsteps);

    int nu;
    int ndual;
    std::size_t i = 0;
    for (i = 0; i < nsteps; i++)
    {
      const StageModel& stage = problem.stages_[i];
      nu = stage.nu();
      ndual = stage.numDual();
      VectorXs feedforward(gains[i].leftCols(1));
      MatrixXs feedback(gains[i].rightCols(stage.ndx1()));

      VectorXs pd_step = feedforward + feedback * workspace.dxs_[i];
      workspace.dus_[i] = pd_step.head(nu);
      workspace.dxs_[i + 1] = pd_step.segment(nu, stage.ndx2());
      workspace.dlams_[i] = pd_step.tail(ndual);
    }

  }

  template<typename Scalar>
  void SolverTpl<Scalar>::computeActiveSetsAndMultipliers(
    const ShootingProblemTpl<Scalar>& problem,
    WorkspaceTpl<Scalar>& workspace,
    ResultsTpl<Scalar>& results) const
  {

  }
  
  template<typename Scalar>
  void SolverTpl<Scalar>::tryStep(
    const ShootingProblemTpl<Scalar>& problem,
    WorkspaceTpl<Scalar>& workspace,
    ResultsTpl<Scalar>& results,
    const Scalar alpha) const
  {
    using StageModel = StageModelTpl<Scalar>;

    const std::size_t nsteps = problem.numSteps();

    std::size_t i = 0;
    for (i = 0; i < nsteps; i++)
    {
      const StageModel& stage = problem.stages_[i];
      stage.xspace1_.integrate(results.xs_[i], alpha * workspace.dxs_[i], workspace.trial_xs_[i]);
      stage.uspace.integrate(results.us_[i], alpha * workspace.dus_[i], workspace.trial_us_[i]);
      workspace.trial_lams_[i] = results.lams_[i] + alpha * workspace.dlams_[i];
    }
    i = nsteps - 1;
    const StageModel& stage_last = problem.stages_[i];
    stage_last.xspace2_.integrate(results.xs_[i], alpha * workspace.dxs_[i], workspace.trial_xs_[i]);
  }

  template<typename Scalar>
  void SolverTpl<Scalar>::backwardPass(
    const ShootingProblemTpl<Scalar>& problem,
    Workspace& workspace,
    ResultsTpl<Scalar>& results) const
  {
    using StageModel = StageModelTpl<Scalar>;
    using StageData = StageDataTpl<Scalar>;

    const ProblemDataTpl<Scalar>& problem_data = *workspace.problem_data;

    const std::size_t nsteps = problem.numSteps();


    /* Terminal node */
    const auto& term_data = *problem_data.term_cost_data;
    value_store_t& term_value = workspace.value_params[nsteps];

    term_value.v_2_ = 2 * term_data->value_;
    term_value.Vx_ = term_data->Lx_;
    term_value.Vxx_ = term_data->Lxx_;

    for (std::size_t step = nsteps - 1; step > 0; step--)
    {
      computeGains(problem, workspace, step);
    }
  }

} // namespace proxddp

