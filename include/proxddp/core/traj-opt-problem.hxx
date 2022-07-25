#pragma once

#include "proxddp/core/traj-opt-problem.hpp"
#include "proxddp/utils/exceptions.hpp"

#include <fmt/core.h>

namespace proxddp {

template <typename Scalar>
TrajOptProblemTpl<Scalar>::TrajOptProblemTpl(
    const VectorXs &x0, const std::vector<shared_ptr<StageModel>> &stages,
    const shared_ptr<CostAbstract> &term_cost)
    : init_state_error(stages[0]->xspace_, stages[0]->nu(), x0),
      stages_(stages), term_cost_(term_cost),
      dummy_term_u0(stages[0]->nu()) {
  dummy_term_u0.setZero();
}

template <typename Scalar>
TrajOptProblemTpl<Scalar>::TrajOptProblemTpl(
    const VectorXs &x0, const int nu,
    const shared_ptr<Manifold> &space,
    const shared_ptr<CostAbstract> &term_cost)
    : TrajOptProblemTpl(InitCstrType(space, nu, x0),
                        nu, term_cost) {}

template <typename Scalar>
void TrajOptProblemTpl<Scalar>::evaluate(const std::vector<VectorXs> &xs,
                                         const std::vector<VectorXs> &us,
                                         TrajOptData &prob_data) const {
  const std::size_t nsteps = numSteps();
  const bool sizes_correct = (xs.size() == nsteps + 1) && (us.size() == nsteps);
  if (!sizes_correct) {
    proxddp_runtime_error(fmt::format(
        "Wrong size for xs or us, expected us.size = {:d}", nsteps));
  }

  init_state_error.evaluate(xs[0], us[0], xs[1], *prob_data.init_data);

  for (std::size_t i = 0; i < nsteps; i++) {
    stages_[i]->evaluate(xs[i], us[i], xs[i + 1], prob_data.getData(i));
  }

  if (term_cost_) {
    term_cost_->evaluate(xs[nsteps], dummy_term_u0, *prob_data.term_cost_data);
  }
  if (term_constraint_) {
    term_constraint_->func_->evaluate(xs[nsteps], dummy_term_u0, xs[nsteps],
                                      *prob_data.term_cstr_data);
  }
}

template <typename Scalar>
void TrajOptProblemTpl<Scalar>::computeDerivatives(
    const std::vector<VectorXs> &xs, const std::vector<VectorXs> &us,
    TrajOptData &prob_data) const {
  const std::size_t nsteps = numSteps();
  const bool sizes_correct = (xs.size() == nsteps + 1) && (us.size() == nsteps);
  if (!sizes_correct) {
    proxddp_runtime_error(fmt::format(
        "Wrong size for xs or us, expected us.size = {:d}", nsteps));
  }

  init_state_error.computeJacobians(xs[0], us[0], xs[1], *prob_data.init_data);

  for (std::size_t i = 0; i < nsteps; i++) {
    stages_[i]->computeDerivatives(xs[i], us[i], xs[i + 1],
                                   prob_data.getData(i));
  }

  if (term_cost_) {
    term_cost_->computeGradients(xs[nsteps], dummy_term_u0,
                                 *prob_data.term_cost_data);
    term_cost_->computeHessians(xs[nsteps], dummy_term_u0,
                                *prob_data.term_cost_data);
  }
  if (term_constraint_) {
    (*term_constraint_)
        .func_->computeJacobians(xs[nsteps], dummy_term_u0, xs[nsteps],
                                 *prob_data.term_cstr_data);
  }
}

template <typename Scalar>
void TrajOptProblemTpl<Scalar>::addStage(const shared_ptr<StageModel> &stage) {
  stages_.push_back(stage);
}

template <typename Scalar>
void TrajOptProblemTpl<Scalar>::setTerminalConstraint(const Constraint &cstr) {
  this->term_constraint_ = cstr;
}

template <typename Scalar>
inline std::size_t TrajOptProblemTpl<Scalar>::numSteps() const {
  return stages_.size();
}

/* TrajOptDataTpl */

template <typename Scalar>
TrajOptDataTpl<Scalar>::TrajOptDataTpl(const TrajOptProblemTpl<Scalar> &problem)
    : init_data(problem.init_state_error.createData()) {
  stage_data.reserve(problem.numSteps());
  for (std::size_t i = 0; i < problem.numSteps(); i++) {
    stage_data.push_back(problem.stages_[i]->createData());
  }

  if (problem.term_cost_) {
    term_cost_data = problem.term_cost_->createData();
  }

  if (problem.term_constraint_) {
    term_cstr_data = (*problem.term_constraint_).func_->createData();
  }
}

} // namespace proxddp
