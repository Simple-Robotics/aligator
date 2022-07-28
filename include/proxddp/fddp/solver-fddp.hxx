#pragma once

#include "proxddp/fddp/solver-fddp.hpp"

namespace proxddp {

template <typename Scalar>
ResultsFDDP<Scalar>::ResultsFDDP(const TrajOptProblemTpl<Scalar> &problem) {
  using StageModel = StageModelTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  const std::size_t nsteps = problem.numSteps();
  xs_.resize(nsteps + 1);
  us_.resize(nsteps);

  xs_default_init(problem, xs_);

  gains_.resize(nsteps);

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &sm = *problem.stages_[i];
    const Manifold &uspace = sm.uspace();

    const int ndx = sm.ndx1();
    const int nu = sm.nu();
    const int ndual = sm.numDual();

    us_[i] = uspace.neutral();
    gains_[i] = MatrixXs::Zero(nu, ndx + 1);
  }
}

template <typename Scalar>
SolverFDDP<Scalar>::SolverFDDP(const Scalar tol, const Scalar reg_init,
                               VerboseLevel verbose)
    : tol_(tol), xreg_(reg_init), ureg_(reg_init), verbose_(verbose) {}

template <typename Scalar>
void SolverFDDP<Scalar>::setup(const Problem &problem) {
  results_ = std::make_unique<Results>(problem);
  workspace_ = std::make_unique<Workspace>(problem);
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
    space.difference(xs[i + 1], workspace.xnexts[i],
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

} // namespace proxddp
