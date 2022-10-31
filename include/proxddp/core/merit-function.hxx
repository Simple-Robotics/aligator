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

  traj_cost_ = computeTrajectoryCost(problem, prob_data);
  Scalar prox_value = computeProxPenalty(workspace, solver->rho());
  penalty_value_ = 0.;
  auto ls_mode = solver->ls_mode;

  bool with_primal_dual_terms = ls_mode == LinesearchMode::PRIMAL_DUAL;

  // initial constraint
  {
    penalty_value_ += .5 * mu() * workspace.lams_plus[0].squaredNorm();
    if (with_primal_dual_terms) {
      penalty_value_ += .5 * dual_weight() * mu() *
                        (workspace.lams_plus[0] - lams[0]).squaredNorm();
    }
  }

  // stage-per-stage
  const std::size_t nsteps = problem.numSteps();
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];
    const StageData &stage_data = prob_data.getStageData(i);

    const ConstraintStack &cstr_mgr = stage.constraints_;
    const std::size_t num_c = cstr_mgr.numConstraints();

    // loop over constraints
    for (std::size_t j = 0; j < num_c; j++) {

      auto lamplus_j =
          cstr_mgr.getSegmentByConstraint(workspace.lams_plus[i + 1], j);
      penalty_value_ += .5 * mu_scaled(j) * lamplus_j.squaredNorm();

      if (with_primal_dual_terms) {
        auto lamin_j = cstr_mgr.getConstSegmentByConstraint(lams[i + 1], j);
        penalty_value_ += .5 * dual_weight() * mu_scaled(j) *
                          (lamplus_j - lamin_j).squaredNorm();
      }
    }
  }

  if (problem.term_constraint_) {

    VectorXs &lamplus = workspace.lams_plus[nsteps + 1];
    penalty_value_ += .5 * mu() * lamplus.squaredNorm();

    if (with_primal_dual_terms) {
      penalty_value_ += .5 * dual_weight() * mu() *
                        (lamplus - lams[nsteps + 1]).squaredNorm();
    }
  }

  value_ = traj_cost_ + prox_value + penalty_value_;
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
    d1 += rho * pdata.Lx_.dot(workspace.dxs[i]);
    if (i < nsteps)
      d1 += rho * pdata.Lu_.dot(workspace.dus[i]);
  }

  // constraints
  {
    const FunctionData &fd = prob_data.getInitData();
    auto &lampdal = workspace.lams_pdal[0];
    d1 += lampdal.dot(fd.Jx_ * workspace.dxs[0]);
  }

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];
    const StageData &stage_data = prob_data.getStageData(i);

    const ConstraintStack &cstr_mgr = stage.constraints_;
    const std::size_t num_c = cstr_mgr.numConstraints();

    auto &lampdal = workspace.lams_pdal[i + 1];
    auto &dx = workspace.dxs[i];
    auto &du = workspace.dus[i];
    auto &dy = workspace.dxs[i + 1];

    for (std::size_t j = 0; j < num_c; j++) {
      const FunctionData &cd = *stage_data.constraint_data[j];
      auto lampdal_j = cstr_mgr.getConstSegmentByConstraint(lampdal, j);

      d1 += lampdal_j.dot(cd.Jx_ * dx);
      d1 += lampdal_j.dot(cd.Ju_ * du);
      d1 += lampdal_j.dot(cd.Jy_ * dy);
    }
  }

  if (problem.term_constraint_) {
    const FunctionData &tcd = prob_data.getTermData();
    auto &lampdal = workspace.lams_pdal[nsteps + 1];
    auto &dx = workspace.dxs.back();

    d1 += lampdal.dot(tcd.Jx_ * dx);
  }

  std::size_t nmul = lams.size();
  for (std::size_t i = 0; i < nmul; i++) {
    auto &laminnr = lams[i];
    auto &lamplus = workspace.lams_plus[i];
    d1 += -mu() * (lamplus - laminnr).dot(workspace.dlams[i]);
  }
  return d1;
}
} // namespace proxddp
