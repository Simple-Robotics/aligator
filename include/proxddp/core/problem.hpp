#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/stage-model.hpp"


namespace proxddp
{
  /// @brief    A problem consists in a succession of nodes.
  template<typename _Scalar>
  struct ShootingProblemTpl
  {
    using Scalar = _Scalar;
    using StageModel = StageModelTpl<Scalar>;
    using StageData = StageDataTpl<Scalar>;

    /// Stages of the control problem.
    std::vector<shared_ptr<StageModel>> stages_;

    inline std::size_t numStages() const { return stages_.size(); }

    ShootingProblemTpl() = default;

    /// @brief Add a stage to the control problem.
    void addStage(const shared_ptr<StageModel>& new_stage);
    /// @copybrief addStage()  
    void addStage(shared_ptr<StageModel>&& new_stage);

    std::vector<shared_ptr<StageData>> createData() const
    {
      std::vector<shared_ptr<StageData>> data_vec;
      data_vec.reserve(stages_.size());
      for (std::size_t i = 0; i < stages_.size(); i++)
      {
        data_vec.push_back(std::move(stages_[i]->createData()));
      }
      return data_vec;
    }

  };
  
} // namespace proxddp

#include "proxddp/core/problem.hxx"
