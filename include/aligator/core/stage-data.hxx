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

} // namespace aligator
