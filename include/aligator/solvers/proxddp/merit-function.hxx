/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
/// @brief Implementation header. Should be included when necessary.
#pragma once

#include "merit-function.hpp"
#include "workspace.hpp"
#include "aligator/core/lagrangian.hpp"
#include "aligator/tracy.hpp"

namespace aligator {

template <typename Scalar>
Scalar costDirectionalDerivative(const WorkspaceTpl<Scalar> &workspace,
                                 const TrajOptDataTpl<Scalar> &prob_data) {
  Scalar d1 = 0.;
  const std::size_t nsteps = workspace.nsteps;
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageDataTpl<Scalar> &sd = *prob_data.stage_data[i];
    const CostDataAbstractTpl<Scalar> &cd = *sd.cost_data;
    d1 += cd.Lx_.dot(workspace.dxs[i]);
    d1 += cd.Lu_.dot(workspace.dus[i]);
  }

  const CostDataAbstractTpl<Scalar> &tcd = *prob_data.term_cost_data;
  d1 += tcd.Lx_.dot(workspace.dxs[nsteps]);
  return d1;
}

// TODO: add missing dual terms
template <typename Scalar>
Scalar ALFunction<Scalar>::evaluate(const Scalar mudyn, const Scalar mucstr,
                                    const TrajOptProblem &problem,
                                    Workspace &workspace) {
  ALIGATOR_TRACY_ZONE_SCOPED;
  TrajOptData &prob_data = workspace.problem_data;
  auto &lams = workspace.lams_plus;
  auto &vs = workspace.vs_plus;
  Scalar penalty_value = 0.;
  bool has_lbdas = !lams.empty();

  auto weighted_norm = [](const auto &v, Scalar m) -> Scalar {
    return m * v.squaredNorm();
  };

  // initial constraint
  if (has_lbdas)
    penalty_value = 0.5 * weighted_norm(lams[0], mucstr);

  // stage-per-stage
  const std::size_t nsteps = problem.numSteps();
  for (std::size_t i = 0; i < nsteps; i++) {
    if (has_lbdas)
      penalty_value += 0.5 * weighted_norm(lams[i + 1], mudyn);
    penalty_value += 0.5 * weighted_norm(vs[i], mucstr);
  }

  if (!problem.term_cstrs_.empty()) {
    penalty_value += 0.5 * weighted_norm(vs[nsteps], mucstr);
  }

  return prob_data.cost_ + penalty_value;
}

template <typename Scalar>
Scalar ALFunction<Scalar>::directionalDerivative(const Scalar mudyn,
                                                 const Scalar mucstr,
                                                 const TrajOptProblem &problem,
                                                 Workspace &workspace) {
  ALIGATOR_TRACY_ZONE_SCOPED;
  const std::size_t nsteps = workspace.nsteps;

  Scalar d1 = 0.;

  const std::vector<VectorXs> &dxs = workspace.dxs;
  const std::vector<VectorXs> &dus = workspace.dus;
  const std::vector<VectorXs> &dvs = workspace.dvs;
  const std::vector<VectorXs> &dlams = workspace.dlams;
  std::vector<VectorXs> &Lxs = workspace.Lxs;
  std::vector<VectorXs> &Lus = workspace.Lus;
  LagrangianDerivatives<Scalar>::compute(problem, workspace.problem_data,
                                         workspace.lams_plus, workspace.vs_plus,
                                         Lxs, Lus);

  assert(dxs.size() == nsteps + 1);
  assert(dus.size() == nsteps);
  assert(dvs.size() == nsteps + 1);
  assert(dlams.size() == nsteps + 1);

  // constraints
  d1 += workspace.Lxs[0].dot(dxs[0]);
  ALIGATOR_RAISE_IF_NAN(d1);

  for (std::size_t i = 0; i < nsteps; i++) {
    d1 += workspace.Lxs[i + 1].dot(dxs[i + 1]);
    ALIGATOR_RAISE_IF_NAN_NAME(dxs[i + 1], fmt::format("dxs[{:d}]", i + 1));
    ALIGATOR_RAISE_IF_NAN(d1);
    d1 += workspace.Lus[i].dot(dus[i]);
    ALIGATOR_RAISE_IF_NAN_NAME(dus[i], fmt::format("dus[{:d}]", i));
    ALIGATOR_RAISE_IF_NAN(d1);
  }

  return d1;
}
} // namespace aligator
