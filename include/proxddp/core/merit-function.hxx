#pragma once

#include "proxddp/core/merit-function.hpp"

namespace proxddp {

template <typename Scalar>
Scalar cost_directional_derivative(const WorkspaceTpl<Scalar> &workspace,
                                   const TrajOptDataTpl<Scalar> &prob_data) {
  Scalar d1 = 0.;
  const std::size_t nsteps = workspace.nsteps;
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageDataTpl<Scalar> &sd = prob_data.getStageData(i);
    const CostDataAbstractTpl<Scalar> &cd = *sd.cost_data;
    d1 += cd.Lx_.dot(workspace.dxs[i]);
    d1 += cd.Lu_.dot(workspace.dus[i]);
  }

  const CostDataAbstractTpl<Scalar> &tcd = *prob_data.term_cost_data;
  d1 += tcd.Lx_.dot(workspace.dxs[nsteps]);
  return d1;
}

template <typename Scalar>
Scalar PDALFunction<Scalar>::evaluate(const SolverType *solver,
                                      const TrajOptProblem &problem,
                                      const std::vector<VectorXs> &lams,
                                      Workspace &workspace) {
  TrajOptData &prob_data = workspace.problem_data;
  Scalar prox_value = 0.;
  if (solver->rho() > 0) {
    prox_value = computeProxPenalty(solver, workspace);
  }
  Scalar penalty_value = 0.;
  auto ls_mode = solver->ls_mode;
  bool use_dual_terms = ls_mode == LinesearchMode::PRIMAL_DUAL;
  const Scalar mu = solver->getLinesearchMu();
  const Scalar dual_weight = solver->dual_weight;
  const std::vector<VectorXs> &lams_plus = workspace.lams_plus;

  // initial constraint
  {
    CstrALWeightStrat weight_strat(mu, false);
    penalty_value += .5 * weight_strat.get(0) * lams_plus[0].squaredNorm();
    if (use_dual_terms) {
      penalty_value += .5 * dual_weight * weight_strat.get(0) *
                       (lams_plus[0] - lams[0]).squaredNorm();
    }
  }

  // local lambda function, defining the op to run on each constraint stack.
  auto execute_on_stack =
      [use_dual_terms = use_dual_terms, dual_weight = dual_weight](
          const ConstraintStack &stack, const VectorXs &lambda,
          const VectorXs &lambda_plus, CstrALWeightStrat &&weight_strat) {
        Scalar r = 0.;
        for (std::size_t k = 0; k < stack.size(); ++k) {
          const auto lamplus_k =
              stack.getConstSegmentByConstraint(lambda_plus, k);
          const auto laminnr_k = stack.getConstSegmentByConstraint(lambda, k);
          r += .5 * weight_strat.get(k) * lamplus_k.squaredNorm();

          if (use_dual_terms) {
            r += .5 * dual_weight * weight_strat.get(k) *
                 (lamplus_k - laminnr_k).squaredNorm();
          }
        }
        return r;
      };

  // stage-per-stage
  const std::size_t nsteps = problem.numSteps();
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];

    const ConstraintStack &cstr_mgr = stage.constraints_;

    penalty_value += execute_on_stack(cstr_mgr, lams[i + 1], lams_plus[i + 1],
                                      CstrALWeightStrat(mu, true));
  }

  if (!problem.term_cstrs_.empty()) {
    assert(lams.size() == nsteps + 2);
    assert(lams_plus.size() == nsteps + 2);
    penalty_value +=
        execute_on_stack(problem.term_cstrs_, lams.back(), lams_plus.back(),
                         CstrALWeightStrat(mu, false));
  }

  return prob_data.cost_ + prox_value + penalty_value;
}

template <typename Scalar>
Scalar PDALFunction<Scalar>::directionalDerivative(
    const SolverType *solver, const TrajOptProblem &problem,
    const std::vector<VectorXs> &lams, Workspace &workspace) {
  TrajOptData &prob_data = workspace.problem_data;
  Scalar d1 = cost_directional_derivative(workspace, prob_data);

  const std::size_t nsteps = workspace.nsteps;
  // prox terms
  const auto &prox_datas = workspace.prox_datas;
  const Scalar rho = solver->rho();
  if (rho > 0) {
    for (std::size_t i = 0; i <= nsteps; i++) {
      const ProximalDataTpl<Scalar> &pdata = *prox_datas[i];
      d1 += rho * pdata.Lx_.dot(workspace.dxs[i]);
      if (i < nsteps)
        d1 += rho * pdata.Lu_.dot(workspace.dus[i]);
    }
  }

  const Scalar mu = solver->getLinesearchMu();
  const auto &dlams = workspace.dlams;
  const auto &lams_plus = workspace.lams_plus;
  const auto &lams_pdal = workspace.lams_pdal;

  // constraints
  {
    const FunctionData &fd = prob_data.getInitData();
    const auto &lampdal = workspace.lams_pdal[0];
    d1 += lampdal.dot(fd.Jx_ * workspace.dxs[0]);
    d1 -= CstrALWeightStrat(mu, false).get(0) *
          (lams_plus[0] - lams[0]).dot(dlams[0]);
  }

  auto execute_on_stack =
      [](const auto &stack, const auto &dx, const auto &du, const auto &dy,
         const auto &dlam, const auto &lam, const auto &lamplus,
         const auto &lampdal,
         const std::vector<shared_ptr<FunctionData>> &constraint_data,
         CstrALWeightStrat &&weight_strat) {
        Scalar r = 0.;
        for (std::size_t k = 0; k < stack.size(); k++) {
          const FunctionData &cd = *constraint_data[k];
          auto lampdal_k = stack.getConstSegmentByConstraint(lampdal, k);
          auto laminnr_k = stack.getConstSegmentByConstraint(lam, k);
          auto lamplus_k = stack.getConstSegmentByConstraint(lamplus, k);
          auto dlam_k = stack.getConstSegmentByConstraint(dlam, k);

          r += lampdal_k.dot(cd.Jx_ * dx);
          r += lampdal_k.dot(cd.Ju_ * du);
          r += lampdal_k.dot(cd.Jy_ * dy);

          r -= weight_strat.get(k) * (lamplus_k - laminnr_k).dot(dlam_k);
        }
        return r;
      };

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];
    const StageData &stage_data = prob_data.getStageData(i);
    const ConstraintStack &cstr_stack = stage.constraints_;

    const auto &dx = workspace.dxs[i];
    const auto &du = workspace.dus[i];
    const auto &dy = workspace.dxs[i + 1];

    d1 += execute_on_stack(cstr_stack, dx, du, dy, dlams[i + 1], lams[i + 1],
                           lams_plus[i + 1], lams_pdal[i + 1],
                           stage_data.constraint_data,
                           CstrALWeightStrat(mu, true));
  }

  const ConstraintStack &term_stack = problem.term_cstrs_;
  for (std::size_t k = 0; k < term_stack.size(); ++k) {
    const FunctionData &tcd = *prob_data.term_cstr_data[k];
    const auto lpdl =
        term_stack.getConstSegmentByConstraint(lams_pdal.back(), k);
    const auto l = term_stack.getConstSegmentByConstraint(lams.back(), k);
    const auto lp = term_stack.getConstSegmentByConstraint(lams_plus.back(), k);
    const auto dl = term_stack.getConstSegmentByConstraint(dlams.back(), k);
    const auto &dx = workspace.dxs.back();

    d1 += lpdl.dot(tcd.Jx_ * dx);
    d1 -= CstrALWeightStrat(mu, false).get(k) * (lp - l).dot(dl);
  }

  return d1;
}

template <typename Scalar>
Scalar PDALFunction<Scalar>::computeProxPenalty(const SolverType *solver,
                                                const Workspace &workspace) {
  Scalar res = 0.;
  const Scalar rho = solver->rho();
  const std::size_t nsteps = workspace.nsteps;
  for (std::size_t i = 0; i <= nsteps; i++) {
    res += rho * workspace.prox_datas[i]->value_;
  }
  return res;
}
} // namespace proxddp
