#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/stage-model.hpp"

#include <fmt/core.h>


namespace proxddp
{
  /// @brief    A problem consists in a succession of nodes.
  template<typename _Scalar>
  struct ShootingProblemTpl
  {
    using Scalar = _Scalar;
    using StageModel = StageModelTpl<Scalar>;
    using StageData = StageDataTpl<Scalar>;

    using ProblemData = ProblemDataTpl<Scalar>;

    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

    /// Stages of the control problem.
    std::vector<StageModel> stages_;
    shared_ptr<CostBaseTpl<Scalar>> term_cost_;

    ShootingProblemTpl() = default;
    ShootingProblemTpl(const std::vector<StageModel>& stages) : stages_(stages) {}

    /// @brief Add a stage to the control problem.
    void addStage(const StageModel& new_stage);
    /// @copybrief addStage()
    void addStage(StageModel&& new_stage);

    inline std::size_t numSteps() const;

    /// @brief Rollout the problem costs, constraints, dynamics, stage per stage.
    void evaluate(const std::vector<VectorXs>& xs,
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

    /// @brief Rollout the problem derivatives, stage per stage.
    void computeDerivatives(const std::vector<VectorXs>& xs,
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

      for (std::size_t i = 0; i < nsteps; i++)
      {
        const StageModel& stage = stages_[i];
        stage.computeDerivatives(xs[i], us[i], xs[i + 1], *prob_data.stage_data[i]);
      }

      if (term_cost_)
      {
        term_cost_->computeGradients(xs[nsteps], us[nsteps - 1], *prob_data.term_cost_data);
        term_cost_->computeHessians(xs[nsteps], us[nsteps - 1], *prob_data.term_cost_data);
      }
    }

    shared_ptr<ProblemData> createData() const
    {
      return std::make_shared<ProblemData>(*this);
    }

  };

  template<typename _Scalar>
  struct ProblemDataTpl
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

    using StageDataPtr = shared_ptr<StageDataTpl<Scalar>>;
    /// Data structs for each stage of the problem.
    std::vector<StageDataPtr> stage_data;
    /// Terminal cost data.
    shared_ptr<CostDataTpl<Scalar>> term_cost_data;

    ProblemDataTpl(const ShootingProblemTpl<Scalar>& problem)
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
  };
  
} // namespace proxddp

#include "proxddp/core/problem.hxx"
