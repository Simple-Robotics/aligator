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

    const ConstraintContainer<Scalar> &cstr_mgr = stage.constraints_;
    const std::size_t num_c = cstr_mgr.numConstraints();
    // loop over constraints
    // get corresponding multipliers from allocated memory
    for (std::size_t j = 0; j < num_c; j++) {
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
    penalty_value += proxnlp::evaluateMoreauEnvelope(*tc.set, c_s_expr,
                                                     lamplus * mu(), mu_inv());
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
    WorkspaceTpl<Scalar> &workspace, TrajOptDataTpl<Scalar> &prob_data) {
  Scalar d1 = costDirectionalDerivative(workspace, prob_data);

  const std::size_t nsteps = workspace.nsteps;
  // prox terms
  const auto &prox_datas = workspace.prox_datas;
  const Scalar rho = solver->rho();
  for (std::size_t i = 0; i <= nsteps; i++) {
    const ProximalDataTpl<Scalar> &pdata = *prox_datas[i];
    d1 += rho * pdata.Lx_.dot(workspace.dxs_[i]);
    if (i < nsteps)
      d1 += rho * pdata.Lu_.dot(workspace.dus_[i]);
  }

  // constraints
  {
    const FunctionData &fd = prob_data.getInitData();
    auto &lampdal = workspace.lams_pdal[0];
    d1 += mu_inv() * lampdal.dot(fd.Jx_ * workspace.dxs_[0]);
    d1 += mu_inv() * lampdal.dot(fd.Ju_ * workspace.dus_[0]);
    d1 += mu_inv() * lampdal.dot(fd.Jy_ * workspace.dxs_[1]);
  }

  for (std::size_t i = 0; i < nsteps; i++) {
    const ConstraintContainer<Scalar> cont = problem.stages_[i]->constraints_;
    const StageData &stage_data = prob_data.getStageData(i);

    const ConstraintContainer<Scalar> &cstr_mgr = stage.constraints_;
    const std::size_t num_c = cstr_mgr.numConstraints();

    auto &lampdal = workspace.lams_pdal[i + 1];
    auto &dx = workspace.dxs_[i];
    auto &du = workspace.dus_[i];
    auto &dy = workspace.dxs_[i + 1];

    for (std::size_t j = 0; j < num_c; j++) {
      const FunctionData &cd = *stage_data.constraint_data[j];
      auto lampdal_j = cstr_mgr.getConstSegmentByConstraint(lampdal, j);

      d1 += mu_inv_scaled() * lampdal_j.dot(cd.Jx_ * dx);
      d1 += mu_inv_scaled() * lampdal_j.dot(cd.Ju_ * du);
      d1 += mu_inv_scaled() * lampdal_j.dot(cd.Jy_ * dy);
    }
  }

  for (std::size_t i = 0; i <= nsteps; i++) {
    auto &laminnr = workspace.trial_lams[i];
    auto &lamplus = workspace.lams_plus[i];
    d1 += mu_inv_scaled() * (lamplus - laminnr).dot(workspace.dlams_[i]);
  }
}
} // namespace proxddp
