/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
/// @brief Template definitions header.
#pragma once

#include "aligator/core/traj-opt-problem.hpp"
#include "aligator/utils/mpc-util.hpp"
#include "aligator/tracy.hpp"

#include <fmt/format.h>

namespace aligator {

template <typename Scalar>
TrajOptProblemTpl<Scalar>::TrajOptProblemTpl(
    xyz::polymorphic<UnaryFunction> init_constraint,
    const std::vector<xyz::polymorphic<StageModel>> &stages,
    xyz::polymorphic<CostAbstract> term_cost)
    : init_constraint_(std::move(init_constraint)), stages_(stages),
      term_cost_(std::move(term_cost)), unone_(term_cost_->nu) {
  unone_.setZero();
  init_state_error_ = dynamic_cast<StateErrorResidual *>(&*init_constraint_);
}

template <typename Scalar>
TrajOptProblemTpl<Scalar>::TrajOptProblemTpl(
    const ConstVectorRef &x0,
    const std::vector<xyz::polymorphic<StageModel>> &stages,
    xyz::polymorphic<CostAbstract> term_cost)
    : TrajOptProblemTpl(
          StateErrorResidual(stages[0]->xspace_, stages[0]->nu(), x0), stages,
          std::move(term_cost)) {
  init_state_error_ = static_cast<StateErrorResidual *>(&*init_constraint_);
}

template <typename Scalar>
TrajOptProblemTpl<Scalar>::TrajOptProblemTpl(
    xyz::polymorphic<UnaryFunction> init_constraint,
    xyz::polymorphic<CostAbstract> term_cost)
    : TrajOptProblemTpl(std::move(init_constraint), {}, std::move(term_cost)) {}

template <typename Scalar>
TrajOptProblemTpl<Scalar>::TrajOptProblemTpl(
    const ConstVectorRef &x0, const int nu, xyz::polymorphic<Manifold> space,
    xyz::polymorphic<CostAbstract> term_cost)
    : TrajOptProblemTpl(StateErrorResidual{std::move(space), nu, x0},
                        std::move(term_cost)) {}

template <typename Scalar>
Scalar TrajOptProblemTpl<Scalar>::evaluate(
    const std::vector<VectorXs> &xs, const std::vector<VectorXs> &us,
    Data &prob_data, ALIGATOR_MAYBE_UNUSED std::size_t num_threads) const {
  ALIGATOR_TRACY_ZONE_SCOPED_N("TrajOptProblem::evaluate");
  const std::size_t nsteps = numSteps();
  if (xs.size() != nsteps + 1)
    ALIGATOR_RUNTIME_ERROR("Wrong size for xs (got {:d}, expected {:d})",
                           xs.size(), nsteps + 1);
  if (us.size() != nsteps)
    ALIGATOR_RUNTIME_ERROR("Wrong size for us (got {:d}, expected {:d})",
                           us.size(), nsteps);

  init_constraint_->evaluate(xs[0], *prob_data.init_data);

  auto &sds = prob_data.stage_data;

  Eigen::setNbThreads(1);
#pragma omp parallel for num_threads(num_threads) schedule(auto)
  for (std::size_t i = 0; i < nsteps; i++) {
    stages_[i]->evaluate(xs[i], us[i], xs[i + 1], *sds[i]);
  }
  Eigen::setNbThreads(0);

  term_cost_->evaluate(xs[nsteps], unone_, *prob_data.term_cost_data);

  for (std::size_t k = 0; k < term_cstrs_.size(); ++k) {
    const auto &func = term_cstrs_.funcs[k];
    auto &td = prob_data.term_cstr_data[k];
    func->evaluate(xs[nsteps], unone_, *td);
  }
  prob_data.cost_ = computeTrajectoryCost(prob_data);
  return prob_data.cost_;
}

template <typename Scalar>
void TrajOptProblemTpl<Scalar>::computeDerivatives(
    const std::vector<VectorXs> &xs, const std::vector<VectorXs> &us,
    Data &prob_data, ALIGATOR_MAYBE_UNUSED std::size_t num_threads,
    bool compute_second_order) const {
  ALIGATOR_TRACY_ZONE_SCOPED_N("TrajOptProblem::computeDerivatives");
  const std::size_t nsteps = numSteps();
  if (xs.size() != nsteps + 1)
    ALIGATOR_RUNTIME_ERROR("Wrong size for xs (got {:d}, expected {:d})",
                           xs.size(), nsteps + 1);
  if (us.size() != nsteps)
    ALIGATOR_RUNTIME_ERROR("Wrong size for us (got {:d}, expected {:d})",
                           us.size(), nsteps);

  init_constraint_->computeJacobians(xs[0], *prob_data.init_data);

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
    const auto &func = term_cstrs_.funcs[k];
    auto &td = prob_data.term_cstr_data[k];
    func->computeJacobians(xs[nsteps], unone_, *td);
  }
}

template <typename Scalar>
void TrajOptProblemTpl<Scalar>::addStage(
    const xyz::polymorphic<StageModel> &stage) {
  stages_.push_back(stage);
}

template <typename Scalar>
bool TrajOptProblemTpl<Scalar>::checkIntegrity() const {
  bool ok = true;

  if (numSteps() == 0)
    return true;

  if (numSteps() > 0) {
    if (stages_[0].valueless_after_move())
      return false;
    ok &= stages_[0]->ndx1() == init_constraint_->ndx1;
  }

  std::size_t k = 1;
  for (; k < numSteps(); k++) {
    if (stages_[k].valueless_after_move())
      return false;
    ok &= stages_[k - 1]->nx2() == stages_[k]->nx1();
    ok &= stages_[k - 1]->ndx2() == stages_[k]->ndx1();
  }
  if (term_cost_.valueless_after_move())
    return false;
  ok &= stages_[k - 1]->nx2() == term_cost_->nx();
  ok &= stages_[k - 1]->ndx2() == term_cost_->ndx();
  return ok;
}

template <typename Scalar>
void TrajOptProblemTpl<Scalar>::addTerminalConstraint(
    const StageConstraintTpl<Scalar> &cstr) {
  term_cstrs_.pushBack(cstr.func, cstr.set);
}

template <typename Scalar>
inline std::size_t TrajOptProblemTpl<Scalar>::numSteps() const {
  return stages_.size();
}

template <typename Scalar>
void TrajOptProblemTpl<Scalar>::replaceStageCircular(
    const xyz::polymorphic<StageModel> &model) {
  addStage(model);
  rotate_vec_left(stages_);
  stages_.pop_back();
}

} // namespace aligator
