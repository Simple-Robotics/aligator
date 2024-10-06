/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/traj-opt-data.hpp"
#include "aligator/core/traj-opt-problem.hpp"
#include "stage-data.hpp"
#include "cost-abstract.hpp"
#include "aligator/tracy.hpp"

namespace aligator {

template <typename Scalar>
TrajOptDataTpl<Scalar>::TrajOptDataTpl(const TrajOptProblemTpl<Scalar> &problem)
    : init_data(problem.init_constraint_->createData()) {
  stage_data.reserve(problem.numSteps());
  for (std::size_t i = 0; i < problem.numSteps(); i++) {
    stage_data.push_back(problem.stages_[i]->createData());
    stage_data[i]->checkData();
  }
  term_cost_data = problem.term_cost_->createData();

  if (!problem.term_cstrs_.empty())
    term_cstr_data.reserve(problem.term_cstrs_.size());
  for (std::size_t k = 0; k < problem.term_cstrs_.size(); k++) {
    const auto &func = problem.term_cstrs_.funcs[k];
    term_cstr_data.push_back(func->createData());
  }
}

template <typename Scalar>
Scalar computeTrajectoryCost(const TrajOptDataTpl<Scalar> &problem_data) {
  ALIGATOR_NOMALLOC_SCOPED;
  ALIGATOR_TRACY_ZONE_SCOPED;
  Scalar traj_cost = 0.;

  const std::size_t nsteps = problem_data.numSteps();
  const auto &sds = problem_data.stage_data;

#pragma omp simd reduction(+ : traj_cost)
  for (std::size_t i = 0; i < nsteps; i++) {
    traj_cost += sds[i]->cost_data->value_;
  }
  traj_cost += problem_data.term_cost_data->value_;

  return traj_cost;
}

} // namespace aligator
