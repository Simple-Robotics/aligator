#pragma once

#include "proxddp/core/traj-opt-problem.hpp"
#include "proxddp/core/stage-data.hpp"
#include "proxddp/utils/exceptions.hpp"
#include "proxddp/utils/mpc-util.hpp"
#include "proxddp/threads.hpp"

#include <fmt/format.h>

namespace proxddp {

template <typename Scalar>
TrajOptProblemTpl<Scalar>::TrajOptProblemTpl(
    shared_ptr<UnaryFunction> init_constraint,
    const std::vector<shared_ptr<StageModel>> &stages,
    shared_ptr<CostAbstract> term_cost)
    : init_condition_(init_constraint), stages_(stages), term_cost_(term_cost),
      unone_(term_cost->nu), num_threads_(1) {
  unone_.setZero();
  checkStages();
  if (auto se =
          std::dynamic_pointer_cast<StateErrorResidual>(init_condition_)) {
    init_state_error_ = se.get();
  }
}

template <typename Scalar>
TrajOptProblemTpl<Scalar>::TrajOptProblemTpl(
    const ConstVectorRef &x0, const std::vector<shared_ptr<StageModel>> &stages,
    shared_ptr<CostAbstract> term_cost)
    : TrajOptProblemTpl(
          createStateError(x0, stages[0]->xspace_, stages[0]->nu()), stages,
          term_cost) {
  init_state_error_ = static_cast<StateErrorResidual *>(init_condition_.get());
}

template <typename Scalar>
TrajOptProblemTpl<Scalar>::TrajOptProblemTpl(
    shared_ptr<UnaryFunction> init_constraint,
    shared_ptr<CostAbstract> term_cost)
    : init_condition_(init_constraint), term_cost_(term_cost),
      unone_(term_cost->nu), init_state_error_(nullptr), num_threads_(1) {
  unone_.setZero();
  if (auto se =
          std::dynamic_pointer_cast<StateErrorResidual>(init_condition_)) {
    init_state_error_ = se.get();
  }
}

template <typename Scalar>
TrajOptProblemTpl<Scalar>::TrajOptProblemTpl(const ConstVectorRef &x0,
                                             const int nu,
                                             shared_ptr<Manifold> space,
                                             shared_ptr<CostAbstract> term_cost)
    : TrajOptProblemTpl(createStateError(x0, space, nu), term_cost) {}

template <typename Scalar>
Scalar TrajOptProblemTpl<Scalar>::evaluate(const std::vector<VectorXs> &xs,
                                           const std::vector<VectorXs> &us,
                                           Data &prob_data) const {
  const std::size_t nsteps = numSteps();
  const bool sizes_correct = (xs.size() == nsteps + 1) && (us.size() == nsteps);
  if (!sizes_correct) {
    PROXDDP_RUNTIME_ERROR(fmt::format(
        "Wrong size for xs or us, expected us.size = {:d}", nsteps));
  }

  init_condition_->evaluate(xs[0], prob_data.getInitData());

  auto &sds = prob_data.stage_data;
  for (std::size_t i = 0; i < nsteps; i++) {
    stages_[i]->evaluate(xs[i], us[i], xs[i + 1], *sds[i]);
  }

  term_cost_->evaluate(xs[nsteps], unone_, *prob_data.term_cost_data);

  for (std::size_t k = 0; k < term_cstrs_.size(); ++k) {
    const ConstraintType &tc = term_cstrs_[k];
    auto &td = prob_data.term_cstr_data[k];
    tc.func->evaluate(xs[nsteps], unone_, xs[nsteps], *td);
  }
  prob_data.cost_ = computeTrajectoryCost(prob_data);
  return prob_data.cost_;
}

template <typename Scalar>
void TrajOptProblemTpl<Scalar>::computeDerivatives(
    const std::vector<VectorXs> &xs, const std::vector<VectorXs> &us,
    Data &prob_data) const {
  const std::size_t nsteps = numSteps();
  const bool sizes_correct = (xs.size() == nsteps + 1) && (us.size() == nsteps);
  if (!sizes_correct) {
    PROXDDP_RUNTIME_ERROR(fmt::format(
        "Wrong size for xs or us, expected us.size = {:d}", nsteps));
  }

  init_condition_->computeJacobians(xs[0], prob_data.getInitData());

  prob_data.xs_copy = xs;
  auto &sds = prob_data.stage_data;

#pragma omp parallel for num_threads(num_threads_)
  for (std::size_t i = 0; i < nsteps; i++) {
    stages_[i]->computeDerivatives(xs[i], us[i], prob_data.xs_copy[i + 1],
                                   *sds[i]);
  }

  if (term_cost_) {
    term_cost_->computeGradients(xs[nsteps], unone_, *prob_data.term_cost_data);
    term_cost_->computeHessians(xs[nsteps], unone_, *prob_data.term_cost_data);
  }

  for (std::size_t k = 0; k < term_cstrs_.size(); ++k) {
    const ConstraintType &tc = term_cstrs_[k];
    auto &td = prob_data.term_cstr_data[k];
    tc.func->computeJacobians(xs[nsteps], unone_, xs[nsteps], *td);
  }
}

template <typename Scalar>
void TrajOptProblemTpl<Scalar>::addStage(const shared_ptr<StageModel> &stage) {
  if (stage == nullptr)
    PROXDDP_RUNTIME_ERROR("Input stage is null.");
  stages_.push_back(stage);
}

template <typename Scalar> void TrajOptProblemTpl<Scalar>::checkStages() const {
  for (auto st = begin(stages_); st != end(stages_); ++st) {
    if (*st == nullptr) {
      long d = std::distance(stages_.begin(), st);
      PROXDDP_RUNTIME_ERROR(fmt::format("Stage {:d} is null.", d));
    }
  }
}

template <typename Scalar>
void TrajOptProblemTpl<Scalar>::setTerminalConstraint(
    const ConstraintType &cstr) {
  removeTerminalConstraints();
  addTerminalConstraint(cstr);
}

template <typename Scalar>
void TrajOptProblemTpl<Scalar>::addTerminalConstraint(
    const ConstraintType &cstr) {
  term_cstrs_.pushBack(cstr);
}

template <typename Scalar>
inline std::size_t TrajOptProblemTpl<Scalar>::numSteps() const {
  return stages_.size();
}

template <typename Scalar>
void TrajOptProblemTpl<Scalar>::replaceStageCircular(
    const shared_ptr<StageModel> &model) {
  addStage(model);
  rotate_vec_left(stages_);
  stages_.pop_back();
}

template <typename Scalar>
Scalar TrajOptProblemTpl<Scalar>::computeTrajectoryCost(
    const Data &problem_data) const {
  PROXDDP_NOMALLOC_BEGIN;
  Scalar traj_cost = 0.;

  const std::size_t nsteps = numSteps();
  const auto &sds = problem_data.stage_data;

#pragma omp simd reduction(+ : traj_cost)
  for (std::size_t i = 0; i < nsteps; i++) {
    traj_cost += sds[i]->cost_data->value_;
  }
  traj_cost += problem_data.term_cost_data->value_;

  PROXDDP_NOMALLOC_END;
  return traj_cost;
}

/* TrajOptDataTpl */

template <typename Scalar>
TrajOptDataTpl<Scalar>::TrajOptDataTpl(const TrajOptProblemTpl<Scalar> &problem)
    : init_data(problem.init_condition_->createData()) {
  stage_data.reserve(problem.numSteps());
  for (std::size_t i = 0; i < problem.numSteps(); i++) {
    stage_data.push_back(problem.stages_[i]->createData());
    stage_data[i]->checkData();
  }

  if (problem.term_cost_) {
    term_cost_data = problem.term_cost_->createData();
  }

  if (!problem.term_cstrs_.empty())
    term_cstr_data.reserve(problem.term_cstrs_.size());
  for (std::size_t k = 0; k < problem.term_cstrs_.size(); k++) {
    const ConstraintType &tc = problem.term_cstrs_[k];
    term_cstr_data.push_back(tc.func->createData());
  }
}

} // namespace proxddp
