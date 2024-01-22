#pragma once

#include "merit-function.hpp"
#include "workspace.hpp"
#include "results.hpp"
#include "aligator/core/lagrangian.hpp"

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

template <typename Scalar>
Scalar PDALFunction<Scalar>::evaluate(const Scalar mu,
                                      const TrajOptProblem &problem,
                                      const std::vector<VectorXs> &lams,
                                      const std::vector<VectorXs> &vs,
                                      Workspace &workspace) {
  TrajOptData &prob_data = workspace.problem_data;
  Scalar penalty_value = 0.;
  const std::vector<VectorXs> &lams_pdal = workspace.lams_pdal;

  // initial constraint
  {
    auto e = 0.5 * mu * lams_pdal[0];
    penalty_value +=
        1. / mu * e.squaredNorm() + 0.25 * mu * lams[0].squaredNorm();
  }

  // local lambda function, defining the op to run on each constraint stack.
  auto execute_on_stack = [](const VectorXs &lambda, const VectorXs &lams_pdal,
                             CstrProximalScaler &weight_strat) {
    auto e1 = weight_strat.apply(lams_pdal);
    auto e2 = weight_strat.apply(lambda);
    Scalar r = 0.25 * e1.dot(lams_pdal);
    r += 0.25 * e2.dot(lambda);
    return r;
  };

  // stage-per-stage
  const std::size_t nsteps = problem.numSteps();
  for (std::size_t i = 0; i < nsteps; i++) {
    penalty_value += execute_on_stack(lams[i + 1], lams_pdal[i + 1],
                                      workspace.cstr_scalers[i]);
  }

  if (!problem.term_cstrs_.empty()) {
    assert(lams.size() == nsteps + 2);
    penalty_value += execute_on_stack(lams.back(), lams_pdal.back(),
                                      workspace.cstr_scalers.back());
  }

  return prob_data.cost_ + penalty_value;
}

template <typename Scalar>
Scalar PDALFunction<Scalar>::directionalDerivative(
    const Scalar mu, const TrajOptProblem &problem,
    const std::vector<VectorXs> &lams, const std::vector<VectorXs> &vs,
    Workspace &workspace) {
  TrajOptData &prob_data = workspace.problem_data;
  const std::size_t nsteps = workspace.nsteps;

  Scalar d1 = costDirectionalDerivative(workspace, prob_data);

  const std::vector<VectorXs> &dxs = workspace.dxs;
  const std::vector<VectorXs> &dus = workspace.dus;
  const std::vector<VectorXs> &dvs = workspace.dvs;
  const std::vector<VectorXs> &dlams = workspace.dlams;
  const std::vector<VectorXs> &vs_pdal = workspace.vs_pdal;
  const std::vector<VectorXs> &lams_pdal = workspace.lams_pdal;
  std::vector<VectorXs> &Lxs = workspace.Lxs_;
  std::vector<VectorXs> &Lus = workspace.Lus_;
  LagrangianDerivatives<Scalar>::compute(problem, workspace.problem_data,
                                         lams_pdal, vs_pdal, Lxs, Lus);

  // constraints
  {
    const auto &lampdal = workspace.lams_pdal[0];
    const auto e = 0.5 * mu * (lams[0] - lampdal);

    d1 += workspace.Lxs_[0].dot(dxs[0]);
    d1 += e.dot(dlams[0]);
  }

  auto execute_on_stack = [](const auto &dlam, const VectorXs &lam,
                             const VectorXs &lampdal,
                             CstrProximalScaler &weight_strat) {
    auto e = 0.5 * weight_strat.apply(lam - lampdal);
    return e.dot(dlam);
  };

  for (std::size_t i = 0; i < nsteps; i++) {
    d1 += workspace.Lxs_[i + 1].dot(dxs[i + 1]);
    d1 += workspace.Lus_[i].dot(dus[i]);
    d1 += execute_on_stack(dlams[i + 1], lams[i + 1], lams_pdal[i + 1],
                           workspace.cstr_scalers[i]);
  }

  d1 += workspace.Lxs_[nsteps].dot(dxs[nsteps]);

  const ConstraintStack &term_stack = problem.term_cstrs_;
  if (!term_stack.empty()) {
    d1 += execute_on_stack(dlams.back(), lams.back(), lams_pdal.back(),
                           workspace.cstr_scalers.back());
  }

  return d1;
}
} // namespace aligator
