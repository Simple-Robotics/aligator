/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/traj-opt-data.hpp"

namespace aligator {

template <typename Scalar>
TrajOptDataTpl<Scalar>::TrajOptDataTpl(const TrajOptProblemTpl<Scalar> &problem)
    : init_condition_common_model_data_container(
          problem.init_condition_common_model_container_.createData()),
      init_data(problem.init_condition_->createData()),
      term_common_model_data_container(
          problem.term_common_model_container_.createData()) {
  stage_data.reserve(problem.numSteps());
  for (std::size_t i = 0; i < problem.numSteps(); i++) {
    stage_data.push_back(problem.stages_[i]->createData());
    stage_data[i]->checkData();
  }

  if (problem.term_cost_) {
    term_cost_data =
        problem.term_cost_->createData(term_common_model_data_container);
  }

  if (!problem.term_cstrs_.empty())
    term_cstr_data.reserve(problem.term_cstrs_.size());
  for (std::size_t k = 0; k < problem.term_cstrs_.size(); k++) {
    const ConstraintType &tc = problem.term_cstrs_[k];
    term_cstr_data.push_back(
        tc.func->createData(term_common_model_data_container));
  }
}

} // namespace aligator
