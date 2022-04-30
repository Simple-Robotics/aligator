#pragma once

#include "proxddp/core/problem.hpp"


namespace proxddp
{
  template<typename Scalar>
  void ShootingProblemTpl<Scalar>::addStage(const shared_ptr<StageModelTpl<Scalar>>& new_stage)
  {
    stages_.push_back(new_stage);
  }

  template<typename Scalar>
  void ShootingProblemTpl<Scalar>::addStage(shared_ptr<StageModelTpl<Scalar>>&& new_stage)
  {
    stages_.push_back(std::move(new_stage));
  }
  
} // namespace proxddp

