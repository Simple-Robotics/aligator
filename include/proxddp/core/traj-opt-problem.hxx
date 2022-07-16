#pragma once

#include "proxddp/core/traj-opt-problem.hpp"

#include <fmt/core.h>
#include <fmt/ostream.h>

namespace proxddp {
template <typename Scalar>
void TrajOptProblemTpl<Scalar>::addStage(const StageModel &stage) {
  stages_.push_back(std::make_shared<StageModel>(stage));
}

template <typename Scalar>
void TrajOptProblemTpl<Scalar>::addStage(StageModel &&stage) {
  stages_.push_back(std::make_shared<StageModel>(stage));
}

template <typename Scalar>
inline std::size_t TrajOptProblemTpl<Scalar>::numSteps() const {
  return stages_.size();
}

template <typename Scalar>
void TrajOptProblemTpl<Scalar>::evaluate(const std::vector<VectorXs> &xs,
                                         const std::vector<VectorXs> &us,
                                         ProblemData &prob_data) const {
  const std::size_t nsteps = numSteps();
  const bool sizes_correct = (xs.size() == nsteps + 1) && (us.size() == nsteps);
  if (!sizes_correct) {
    throw std::runtime_error(fmt::format(
        "Wrong size for xs or us, expected us.size = {:d}", nsteps));
  }

  init_state_error.evaluate(xs[0], us[0], xs[1], *prob_data.init_data);

  for (std::size_t i = 0; i < nsteps; i++) {
    stages_[i]->evaluate(xs[i], us[i], xs[i + 1], prob_data.stage_data[i]);
  }

  if (term_cost_) {
    term_cost_->evaluate(xs[nsteps], us[nsteps - 1], *prob_data.term_cost_data);
  }
  if (term_constraint_) {
    term_constraint_->func_->evaluate(xs[nsteps], us[nsteps - 1], xs[nsteps],
                                      *prob_data.term_cstr_data);
  }
}

template <typename Scalar>
void TrajOptProblemTpl<Scalar>::computeDerivatives(
    const std::vector<VectorXs> &xs, const std::vector<VectorXs> &us,
    ProblemData &prob_data) const {
  const std::size_t nsteps = numSteps();
  const bool sizes_correct = (xs.size() == nsteps + 1) && (us.size() == nsteps);
  if (!sizes_correct) {
    throw std::runtime_error(fmt::format(
        "Wrong size for xs or us, expected us.size = {:d}", nsteps));
  }

  init_state_error.computeJacobians(xs[0], us[0], xs[1], *prob_data.init_data);

  for (std::size_t i = 0; i < nsteps; i++) {
    stages_[i]->computeDerivatives(xs[i], us[i], xs[i + 1],
                                   prob_data.stage_data[i]);
  }

  if (term_cost_) {
    term_cost_->computeGradients(xs[nsteps], us[nsteps - 1],
                                 *prob_data.term_cost_data);
    term_cost_->computeHessians(xs[nsteps], us[nsteps - 1],
                                *prob_data.term_cost_data);
  }
  if (term_constraint_) {
    (*term_constraint_)
        .func_->computeJacobians(xs[nsteps], us[nsteps - 1], xs[nsteps],
                                 *prob_data.term_cstr_data);
  }
}

template <typename Scalar>
TrajOptDataTpl<Scalar>::TrajOptDataTpl(const TrajOptProblemTpl<Scalar> &problem)
    : init_data(std::move(problem.init_state_error.createData())) {
  stage_data.reserve(problem.numSteps());
  for (std::size_t i = 0; i < problem.numSteps(); i++) {
    stage_data.push_back(std::move(*problem.stages_[i]->createData()));
  }

  if (problem.term_cost_) {
    term_cost_data = problem.term_cost_->createData();
  }

  if (problem.term_constraint_) {
    term_cstr_data = (*problem.term_constraint_).func_->createData();
  }
}

} // namespace proxddp
