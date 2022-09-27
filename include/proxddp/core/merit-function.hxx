#pragma once

namespace proxddp {

template <typename Scalar>
PDALFunction<Scalar>::PDALFunction(SolverProxDDP<Scalar> const *solver)
    : solver(solver) {}

template <typename Scalar>
Scalar PDALFunction<Scalar>::evaluate(const TrajOptProblemTpl<Scalar> &problem,
                                      const std::vector<VectorXs> &lams,
                                      WorkspaceTpl<Scalar> &workspace,
                                      TrajOptDataTpl<Scalar> &prob_data) {

  traj_cost = computeTrajectoryCost(problem, prob_data);
  prox_value = computeProxPenalty(workspace, solver->rho());
  penalty_value = 0.;
  auto ls_mode = solver->ls_mode;

  bool with_primal_dual_terms = ls_mode == LinesearchMode::PRIMAL_DUAL;

  // initial constraint
  {
    penalty_value += .5 * mu() * workspace.lams_plus[0].squaredNorm();
    if (with_primal_dual_terms) {
      penalty_value += .5 * dual_weight_ * mu() *
                       (workspace.lams_plus[0] - lams[0]).squaredNorm();
    }
  }

  // stage-per-stage
  const std::size_t nsteps = problem.numSteps();
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];
    const StageData &stage_data = prob_data.getStageData(i);

    const std::size_t num_c = stage.numConstraints();
    // loop over constraints
    // get corresponding multipliers from allocated memory
    for (std::size_t j = 0; j < num_c; j++) {
      const ConstraintContainer<Scalar> &cstr_mgr = stage.constraints_;
      const CstrSet &cstr_set = cstr_mgr.getConstraintSet(j);
      const FunctionData &cstr_data = *stage_data.constraint_data[j];
      auto lamplus_j =
          cstr_mgr.getSegmentByConstraint(workspace.lams_plus[i + 1], j);
      auto lamprev_j =
          cstr_mgr.getConstSegmentByConstraint(workspace.prev_lams[i + 1], j);
      auto c_s_expr = cstr_data.value_ + mu_scaled() * lamprev_j;
      penalty_value += proxnlp::evaluateMoreauEnvelope(
          cstr_set, c_s_expr, lamplus_j * mu_scaled(), mu_inv_scaled());
    }
    if (with_primal_dual_terms) {
      penalty_value += .5 * dual_weight_ * mu_scaled() *
                       (workspace.lams_plus[i + 1] - lams[i + 1]).squaredNorm();
    }
  }

  if (problem.term_constraint_) {
    const StageConstraintTpl<Scalar> &tc = *problem.term_constraint_;
    const FunctionData &cstr_data = *prob_data.term_cstr_data;
    VectorXs &lamplus = workspace.lams_plus[nsteps + 1];
    auto c_s_expr = cstr_data.value_ + mu() * workspace.prev_lams[nsteps + 1];
    penalty_value += proxnlp::evaluateMoreauEnvelope(
        *tc.set, c_s_expr, lamplus * mu(), solver->mu_inv());
    if (with_primal_dual_terms) {
      penalty_value +=
          .5 * dual_weight_ * mu() * (lamplus - lams[nsteps + 1]).squaredNorm();
    }
  }

  value_ = traj_cost + prox_value + penalty_value;
  return value_;
}
template <typename Scalar>
Scalar PDALFunction<Scalar>::directionalDerivative(
    const TrajOptProblemTpl<Scalar> &problem, const std::vector<VectorXs> &lams,
    WorkspaceTpl<Scalar> &workspace, TrajOptDataTpl<Scalar> &prob_data) {}
} // namespace proxddp
