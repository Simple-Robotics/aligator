#pragma once

#include "proxddp/core/problem.hpp"


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

} // namespace proxddp

