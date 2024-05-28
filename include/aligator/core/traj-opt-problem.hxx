/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/traj-opt-problem.hpp"
#include "aligator/utils/mpc-util.hpp"
#include "tracy/Tracy.hpp"

#include <fmt/format.h>

namespace aligator {

template <typename Scalar>
TrajOptProblemTpl<Scalar>::TrajOptProblemTpl(
    shared_ptr<UnaryFunction> init_constraint,
    const std::vector<shared_ptr<StageModel>> &stages,
    xyz::polymorphic<CostAbstract> term_cost)
    : init_condition_(init_constraint), stages_(stages), term_cost_(term_cost),
      unone_(term_cost->nu) {
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
    xyz::polymorphic<CostAbstract> term_cost)
    : TrajOptProblemTpl(
          createStateError(x0, stages[0]->xspace_, stages[0]->nu()), stages,
          term_cost) {
  init_state_error_ = static_cast<StateErrorResidual *>(init_condition_.get());
}

template <typename Scalar>
TrajOptProblemTpl<Scalar>::TrajOptProblemTpl(
    shared_ptr<UnaryFunction> init_constraint,
    xyz::polymorphic<CostAbstract> term_cost)
    : init_condition_(init_constraint), term_cost_(term_cost),
      unone_(term_cost->nu),
      init_state_error_(
          dynamic_cast<StateErrorResidual *>(init_condition_.get())) {
  unone_.setZero();
}

template <typename Scalar>
TrajOptProblemTpl<Scalar>::TrajOptProblemTpl(
    const ConstVectorRef &x0, const int nu, xyz::polymorphic<Manifold> space,
    xyz::polymorphic<CostAbstract> term_cost)
    : TrajOptProblemTpl(createStateError(x0, space, nu), term_cost) {}

template <typename Scalar>
Scalar TrajOptProblemTpl<Scalar>::evaluate(
    const std::vector<VectorXs> &xs, const std::vector<VectorXs> &us,
    Data &prob_data, ALIGATOR_MAYBE_UNUSED std::size_t num_threads) const {
  ZoneScopedN("TrajOptProblem::evaluate");
  const std::size_t nsteps = numSteps();
  if (xs.size() != nsteps + 1)
    ALIGATOR_RUNTIME_ERROR(fmt::format(
        "Wrong size for xs (got {:d}, expected {:d})", xs.size(), nsteps + 1));
  if (us.size() != nsteps)
    ALIGATOR_RUNTIME_ERROR(fmt::format(
        "Wrong size for us (got {:d}, expected {:d})", us.size(), nsteps));

  init_condition_->evaluate(xs[0], *prob_data.init_data);

  auto &sds = prob_data.stage_data;

  Eigen::setNbThreads(1);
#pragma omp parallel for num_threads(num_threads) schedule(auto)
  for (std::size_t i = 0; i < nsteps; i++) {
    stages_[i]->evaluate(xs[i], us[i], xs[i + 1], *sds[i]);
  }
  Eigen::setNbThreads(0);

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
    Data &prob_data, ALIGATOR_MAYBE_UNUSED std::size_t num_threads,
    bool compute_second_order) const {
  ZoneScopedN("TrajOptProblem::computeDerivatives");
  const std::size_t nsteps = numSteps();
  if (xs.size() != nsteps + 1)
    ALIGATOR_RUNTIME_ERROR(fmt::format(
        "Wrong size for xs (got {:d}, expected {:d})", xs.size(), nsteps + 1));
  if (us.size() != nsteps)
    ALIGATOR_RUNTIME_ERROR(fmt::format(
        "Wrong size for us (got {:d}, expected {:d})", us.size(), nsteps));

  init_condition_->computeJacobians(xs[0], *prob_data.init_data);

  auto &sds = prob_data.stage_data;

  Eigen::setNbThreads(1);
#pragma omp parallel for num_threads(num_threads) schedule(auto)
  for (std::size_t i = 0; i < nsteps; i++) {
    stages_[i]->computeFirstOrderDerivatives(xs[i], us[i], xs[i + 1], *sds[i]);
    if (compute_second_order) {
      stages_[i]->computeSecondOrderDerivatives(xs[i], us[i], *sds[i]);
    }
  }
  Eigen::setNbThreads(0);

  term_cost_->computeGradients(xs[nsteps], unone_, *prob_data.term_cost_data);
  if (compute_second_order) {
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
    ALIGATOR_RUNTIME_ERROR("Input stage is null.");
  stages_.push_back(stage);
}

template <typename Scalar> void TrajOptProblemTpl<Scalar>::checkStages() const {
  for (auto st = begin(stages_); st != end(stages_); ++st) {
    if (*st == nullptr) {
      long d = std::distance(stages_.begin(), st);
      ALIGATOR_RUNTIME_ERROR(fmt::format("Stage {:d} is null.", d));
    }
  }
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
  ALIGATOR_NOMALLOC_SCOPED;
  ZoneScoped;
  Scalar traj_cost = 0.;

  const std::size_t nsteps = numSteps();
  const auto &sds = problem_data.stage_data;

#pragma omp simd reduction(+ : traj_cost)
  for (std::size_t i = 0; i < nsteps; i++) {
    traj_cost += sds[i]->cost_data->value_;
  }
  traj_cost += problem_data.term_cost_data->value_;

  return traj_cost;
}

} // namespace aligator
