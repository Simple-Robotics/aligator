#pragma once

#include <crocoddyl/core/optctrl/shooting.hpp>
#include "proxddp/core/traj-opt-problem.hpp"
#include "proxddp/compat/crocoddyl/cost-wrap.hpp"
#include "proxddp/compat/crocoddyl/action-model.hpp"

namespace proxddp {
namespace compat {
namespace croc {

template <typename Scalar>
TrajOptProblemTpl<Scalar> convertCrocoddylProblem(
    const boost::shared_ptr<crocoddyl::ShootingProblemTpl<Scalar>>
        &croc_problem) {
  const auto &cpb = *croc_problem;
  using VectorXs = typename math_types<Scalar>::VectorXs;
  using MatrixXs = typename math_types<Scalar>::MatrixXs;
  using StageModel = StageModelTpl<Scalar>;
  using ActionModelWrapper = CrocActionModelWrapperTpl<Scalar>;
  using CrocActModel = crocoddyl::ActionModelAbstractTpl<Scalar>;

  const std::size_t nsteps = cpb.get_T();
  const VectorXs &x0 = cpb.get_x0();

  const auto &running_models = cpb.get_runningModels();

  std::vector<shared_ptr<StageModel>> stages;
  stages.reserve(nsteps);
  for (std::size_t i = 0; i < nsteps; i++) {
    stages.push_back(std::make_shared<ActionModelWrapper>(running_models[i]));
  }
  auto converted_cost = std::make_shared<CrocCostModelWrapperTpl<Scalar>>(
      cpb.get_terminalModel());
  TrajOptProblemTpl<Scalar> problem(x0, stages, converted_cost);
  return problem;
}

} // namespace croc
} // namespace compat
} // namespace proxddp
