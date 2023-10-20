/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/stage-data.hpp"

#include "aligator/core/stage-model.hpp"

namespace aligator {

/* StageDataTpl */

template <typename Scalar>
StageDataTpl<Scalar>::StageDataTpl(const StageModel &stage_model)
    : constraint_data(stage_model.numConstraints()),
      cost_data(stage_model.cost_->createData()) {
  using Function = StageFunctionTpl<Scalar>;
  const std::size_t nc = stage_model.numConstraints();
  constraint_data.reserve(nc);
  for (std::size_t j = 0; j < nc; j++) {
    const shared_ptr<Function> &func = stage_model.constraints_[j].func;
    constraint_data[j] = func->createData();
  }
  dynamics_data = std::dynamic_pointer_cast<DynamicsData>(constraint_data[0]);
}

template <typename Scalar> void StageDataTpl<Scalar>::checkData() {
  const char msg[] = "StageData integrity check failed.";
  if (constraint_data.size() == 0)
    ALIGATOR_RUNTIME_ERROR(fmt::format("{} (constraint_data empty)", msg));
  if (cost_data == 0)
    ALIGATOR_RUNTIME_ERROR(fmt::format("{} (cost_data is nullptr)", msg));
  if (dynamics_data == nullptr)
    ALIGATOR_RUNTIME_ERROR(
        fmt::format("{} (constraint_data[0] should be dynamics data)", msg));
}

} // namespace aligator
