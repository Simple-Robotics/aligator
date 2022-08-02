#pragma once

#include "proxddp/fddp/solver-fddp.hpp"

namespace proxddp {

/* SolverFDDP<Scalar> */

template <typename Scalar>
SolverFDDP<Scalar>::SolverFDDP(const Scalar tol, VerboseLevel verbose,
                               const Scalar reg_init)
    : tol_(tol), xreg_(reg_init), ureg_(reg_init), verbose_(verbose) {}

template <typename Scalar>
void SolverFDDP<Scalar>::setup(const Problem &problem) {
  results_ = std::make_unique<Results>(problem);
  workspace_ = std::make_unique<Workspace>(problem);
  // check if there are any constraints other than dynamics and throw a warning
  std::vector<std::size_t> idx_where_constraints;
  for (std::size_t i = 0; i < problem.numSteps(); i++) {
    const shared_ptr<StageModel> &sm = problem.stages_[i];
    if (sm->numConstraints() > 1) {
      idx_where_constraints.push_back(i);
    }
  }
  if (idx_where_constraints.size() > 0) {
    proxddp_fddp_warning(
        fmt::format("problem stages [{}] have constraints, "
                    "which this solver cannot handle. "
                    "Please use a penalized cost formulation.\n",
                    fmt::join(idx_where_constraints, ", ")));
  }
  if (problem.term_constraint_) {
    proxddp_fddp_warning(
        "problem has a terminal constraint, which this solver cannot "
        "handle.\n");
  }
}

template <typename Scalar>
void SolverFDDP<Scalar>::evaluateGaps(const Problem &problem,
                                      const std::vector<VectorXs> &xs,
                                      const std::vector<VectorXs> &us,
                                      const Workspace &workspace,
                                      Results &results) const {
  const std::size_t nsteps = problem.numSteps();
  const ProblemData &pd = workspace.problem_data;

  const Manifold &space = problem.stages_[0]->xspace();
  space.difference(xs[0], problem.getInitState(), workspace.feas_gaps_[0]);

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &sm = *problem.stages_[i];
    const StageData &sd = pd.getData(i);
    const Manifold &space = sm.xspace();
    space.difference(xs[i + 1], workspace.xnexts_[i],
                     workspace.feas_gaps_[i + 1]);
  }
}

template <typename Scalar>
Scalar SolverFDDP<Scalar>::tryStep(const Problem &problem,
                                   const Results &results, Workspace &workspace,
                                   const Scalar alpha) const {
  forwardPass(problem, results, workspace, alpha);
  problem.evaluate(workspace.trial_xs_, workspace.trial_us_,
                   workspace.trial_prob_data);
  return computeTrajectoryCost(problem, workspace.trial_prob_data);
}

template <typename Scalar>
void SolverFDDP<Scalar>::forwardPass(const Problem &problem,
                                     const Results &results,
                                     Workspace &workspace, const Scalar alpha) {
  const std::size_t nsteps = workspace.nsteps;
  std::vector<VectorXs> &xs_try = workspace.trial_xs_;
  std::vector<VectorXs> &us_try = workspace.trial_us_;
  ProblemData &pd = workspace.problem_data;

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &sm = *problem.stages_[i];
    const DynamicsModelTpl<Scalar> &dm = sm.dyn_model();
    const Manifold &space = sm.xspace();
    StageData &sd = pd.getData(i);
    DynamicsDataTpl<Scalar> &dd = stage_get_dynamics_data(sd);

    VectorXs &dx = workspace.dxs_[i];
    ConstVectorRef ff = results.getFeedforward(i);
    ConstMatrixRef fb = results.getFeedback(i);

    space.difference(results.xs_[i], xs_try[i], dx);
    us_try[i] = results.us_[i] + alpha * ff + fb * dx;
    forwardDynamics(space, dm, xs_try[i], us_try[i], dd, workspace.xnexts_[i]);

    space.integrate(workspace.xnexts_[i],
                    workspace.feas_gaps_[i + 1] * (alpha - 1.), xs_try[i + 1]);
  }
}

template <typename Scalar>
void SolverFDDP<Scalar>::computeInfeasibility(const Problem &problem,
                                              Results &results,
                                              Workspace &workspace) {
  const std::size_t nsteps = workspace.nsteps;
  const ProblemData &pd = workspace.problem_data;
  std::vector<VectorXs> &xs = results.xs_;
  const Manifold &space = problem.stages_[0]->xspace();
  const VectorXs &x0 = problem.getInitState();

  space.difference(xs[0], x0, workspace.feas_gaps_[0]);
  for (std::size_t i = 0; i < nsteps; i++) {
    const Manifold &space = problem.stages_[i]->xspace();
    space.difference(xs[i + 1], workspace.xnexts_[i],
                     workspace.feas_gaps_[i + 1]);
  }

  results.primal_infeasibility = math::infty_norm(workspace.feas_gaps_);
}
template <typename Scalar>
void SolverFDDP<Scalar>::computeCriterion(Workspace &workspace,
                                          Results &results) {
  const std::size_t nsteps = workspace.nsteps;
  std::vector<ConstVectorRef> Qus;
  for (std::size_t i = 0; i < nsteps; i++) {
    Qus.push_back(workspace.q_params[i].Qu_);
  }
  results.dual_infeasibility = math::infty_norm(Qus);
}
} // namespace proxddp
