#pragma once

#include "proxddp/core/shooting-problem.hpp"

#include <fmt/core.h>
#include <fmt/ostream.h>

namespace proxddp
{
  template<typename Scalar>
  void ShootingProblemTpl<Scalar>::addStage(const StageModel& new_stage)
  {
    stages_.push_back(new_stage);
  }

  template<typename Scalar>
  void ShootingProblemTpl<Scalar>::addStage(StageModel&& new_stage)
  {
    stages_.push_back(std::move(new_stage));
  }
  
  template<typename Scalar>
  inline std::size_t ShootingProblemTpl<Scalar>::numSteps() const
  {
    return stages_.size();
  }

  template<typename Scalar>
  void ShootingProblemTpl<Scalar>::
  evaluate(const std::vector<VectorXs>& xs,
           const std::vector<VectorXs>& us,
           ProblemData& prob_data) const
  {
    const std::size_t nsteps = numSteps();
    const bool sizes_correct = (xs.size() == nsteps + 1) && (us.size() == nsteps);
    if (!sizes_correct)
    {
      throw std::runtime_error(
        fmt::format("Wrong size for xs or us, expected us.size = {:d}", nsteps));
    }

    init_state_error.evaluate(xs[0], us[0], xs[1], *prob_data.init_data);

    for (std::size_t i = 0; i < nsteps; i++)
    {
      const StageModel& stage = stages_[i];
      stage.evaluate(xs[i], us[i], xs[i + 1], *prob_data.stage_data[i]);
    }

    if (term_cost_)
    {
      term_cost_->evaluate(xs[nsteps], us[nsteps - 1], *prob_data.term_cost_data);
    }
  }

  template<typename Scalar>
  void ShootingProblemTpl<Scalar>::
  computeDerivatives(const std::vector<VectorXs>& xs,
                     const std::vector<VectorXs>& us,
                     ProblemData& prob_data) const
  {
    const std::size_t nsteps = numSteps();
    auto stage_data = prob_data.stage_data;
    const bool sizes_correct = (xs.size() == nsteps + 1) && (us.size() == nsteps);
    if (!sizes_correct)
    {
      throw std::runtime_error(
        fmt::format("Wrong size for xs or us, expected us.size = {:d}", nsteps));
    }

    init_state_error.computeJacobians(xs[0], us[0], xs[1], *prob_data.init_data);

    for (std::size_t i = 0; i < nsteps; i++)
    {
      const StageModel& stage = stages_[i];
      stage.computeDerivatives(xs[i], us[i], xs[i + 1], *stage_data[i]);
    }

    if (term_cost_)
    {
      term_cost_->computeGradients(xs[nsteps], us[nsteps - 1], *prob_data.term_cost_data);
      term_cost_->computeHessians(xs[nsteps], us[nsteps - 1], *prob_data.term_cost_data);
    }
  }

  template<typename Scalar>
  ShootingProblemDataTpl<Scalar>::
  ShootingProblemDataTpl(const ShootingProblemTpl<Scalar>& problem)
    : init_data(std::move(problem.init_state_error.createData()))
  {
    stage_data.reserve(problem.numSteps());
    for (std::size_t i = 0; i < problem.numSteps(); i++)
    {
      stage_data.push_back(std::move(problem.stages_[i].createData()));
    }

    if (problem.term_cost_)
    {
      term_cost_data = problem.term_cost_->createData();
    }
  }

} // namespace proxddp

