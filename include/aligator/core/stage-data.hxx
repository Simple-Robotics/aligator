#pragma once

#include "stage-data.hpp"
#include "stage-model.hpp"
#include "cost-abstract.hpp"

namespace aligator {

template <typename Scalar>
StageDataTpl<Scalar>::StageDataTpl(const StageModel &stage_model)
    : constraint_data(stage_model.numConstraints())
    , cost_data(stage_model.cost_->createData())
    , dynamics_data(stage_model.dynamics_->createData()) {
  const std::size_t nc = stage_model.numConstraints();

  for (std::size_t j = 0; j < nc; j++) {
    const auto &func = stage_model.constraints_.funcs[j];
    constraint_data[j] = func->createData();
  }
}

template <typename Scalar> void StageDataTpl<Scalar>::checkData() {
  const char msg[] = "StageData integrity check failed.";
  if (cost_data == nullptr)
    ALIGATOR_RUNTIME_ERROR("{} (cost_data cannot be nullptr)", msg);
  if (dynamics_data == nullptr)
    ALIGATOR_RUNTIME_ERROR("{} (dynamics_data cannot be nullptr)", msg);
}

} // namespace aligator
