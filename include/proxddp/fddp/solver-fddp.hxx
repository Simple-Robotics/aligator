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
    forwardDynamics(dm, xs_try[i], us_try[i], dd, workspace.xnexts_[i + 1]);

    space.integrate(workspace.xnexts_[i],
                    workspace.feas_gaps_[i + 1] * (alpha - 1.), xs_try[i + 1]);
  }
}

template <typename Scalar>
Scalar SolverFDDP<Scalar>::tryStep(const Problem &problem,
                                   const Results &results, Workspace &workspace,
                                   const Scalar alpha) {
  forwardPass(problem, results, workspace, alpha);
  problem.evaluate(workspace.trial_xs_, workspace.trial_us_,
                   workspace.trial_prob_data);
  return computeTrajectoryCost(problem, workspace.trial_prob_data);
}

template <typename Scalar>
void SolverFDDP<Scalar>::computeDirectionalDerivatives(Workspace &workspace,
                                                       Results &results,
                                                       Scalar &d1,
                                                       Scalar &d2) const {
  const std::size_t nsteps = workspace.nsteps;
  d1 = 0.; // cost directional derivative
  d2 = 0.;

  assert(workspace.q_params.size() == nsteps);
  assert(workspace.value_params.size() == (nsteps + 1));
  for (std::size_t i = 0; i < nsteps; i++) {
    const QParams &qpar = workspace.q_params[i];
    ConstVectorRef Qu = qpar.Qu_;
    ConstVectorRef ff = results.getFeedforward(i);
    d1 += Qu.dot(ff);
    d2 += ff.dot(workspace.Quuks_[i]);
  }
  for (std::size_t i = 0; i <= nsteps; i++) {
    // account for infeasibility
    const VParams &vpar = workspace.value_params[i];
    VectorXs &ftVxx = workspace.f_t_Vxx_[i];
    ftVxx = vpar.Vxx_ * workspace.feas_gaps_[i];
    d1 += vpar.Vx_.dot(workspace.feas_gaps_[i]);
    d2 = d2 - ftVxx.dot(workspace.feas_gaps_[i]);
  }
}

template <typename Scalar>
void SolverFDDP<Scalar>::directionalDerivativeCorrection(const Problem &problem,
                                                         Workspace &workspace,
                                                         Results &results,
                                                         Scalar &d1,
                                                         Scalar &d2) {
  const std::size_t nsteps = workspace.nsteps;
  const VectorOfVectors &xs = results.xs_;
  const VectorOfVectors &us = results.us_;

  Scalar dv = 0.;
  for (std::size_t i = 0; i < nsteps; i++) {
    const Manifold &space = problem.stages_[i]->xspace();
    space.difference(workspace.trial_xs_[i], xs[i], workspace.dxs_[i]);

    const VParams &vpar = workspace.value_params[i];
    const VectorXs &ftVxx = workspace.f_t_Vxx_[i];
    // ftVxx = vpar.Vxx_ * workspace.feas_gaps_[i]; // same as l.145
    dv += workspace.dxs_[i].dot(ftVxx);
  }

  d1 += dv;
  d2 += -2 * dv;
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

template <typename Scalar>
void SolverFDDP<Scalar>::backwardPass(const Problem &problem,
                                      Workspace &workspace,
                                      Results &results) const {

  const std::size_t nsteps = workspace.nsteps;

  ProblemData &prob_data = workspace.problem_data;
  const CostData &term_cost_data = *prob_data.term_cost_data;
  VParams &term_value = workspace.value_params[nsteps];
  term_value.v_2() = 2 * term_cost_data.value_;
  term_value.Vx_ = term_cost_data.Lx_;
  term_value.Vxx_ = term_cost_data.Lxx_;
  term_value.Vxx_.diagonal().array() += xreg_;
  term_value.storage =
      term_value.storage.template selfadjointView<Eigen::Lower>();

  std::size_t i;
  for (std::size_t k = 0; k < nsteps; k++) {
    i = nsteps - k - 1;
    const VParams &vnext = workspace.value_params[i + 1];
    QParams &qparam = workspace.q_params[i];

    StageModel &sm = *problem.stages_[i];
    StageData &sd = prob_data.getData(i);

    const int nu = sm.nu();
    const int ndx1 = sm.ndx1();
    const int nt = ndx1 + nu;
    assert((qparam.storage.cols() == nt + 1) &&
           (qparam.storage.rows() == nt + 1));
    assert(qparam.grad_.size() == nt);

    const CostData &cd = *sd.cost_data;
    DynamicsDataTpl<Scalar> &dd = stage_get_dynamics_data(sd);

    /* Assemble Q-function */
    ConstMatrixRef J_x_u = dd.jac_buffer_.leftCols(ndx1 + nu);

    qparam.q_2() = 2 * cd.value_;
    qparam.grad_ = cd.grad_;
    qparam.hess_ = cd.hess_;

    // fmt::print("==== NODE t = {:d} ====\n", i);
    // fmt::print("vnext: {}\n", vnext);
    // fmt::print("vgrad: {}\n", vnext.Vx_.transpose());
    // TODO: implement second-order derivatives for the Q-function
    qparam.grad_.noalias() += J_x_u.transpose() * vnext.Vx_;
    qparam.hess_.noalias() += J_x_u.transpose() * vnext.Vxx_ * J_x_u;

    qparam.Quu_.diagonal().array() += ureg_;
    qparam.storage = qparam.storage.template selfadjointView<Eigen::Lower>();

    /* Compute gains */
    // MatrixXs &kkt_mat = workspace.kkt_matrix_bufs[i];
    MatrixXs &kkt_rhs = workspace.kkt_rhs_bufs[i];

    // kkt_mat = qparam.Quu_;
    VectorRef ffwd = results.getFeedforward(i);
    MatrixRef fback = results.getFeedback(i);
    ffwd = -qparam.Qu_;
    fback = -qparam.Qxu_.transpose();
    kkt_rhs = results.gains_[i];

    Eigen::LLT<MatrixXs> &llt = workspace.llts_[i];
    llt.compute(qparam.Quu_);
    llt.solveInPlace(results.gains_[i]);

    workspace.Quuks_[i] = qparam.Quu_ * ffwd;

    /* Compute value function */
    VParams &vcur = workspace.value_params[i];
    vcur.Vx_ = qparam.Qx_ + fback.transpose() * qparam.Qu_;
    vcur.Vxx_ = qparam.Qxx_ + qparam.Qxu_ * fback;
    vcur.Vx_.noalias() += vcur.Vxx_ * workspace.feas_gaps_[i + 1];
    vcur.Vxx_.diagonal().array() += xreg_;
    vcur.storage = vcur.storage.template selfadjointView<Eigen::Lower>();
  }
  assert(i == 0);
}

} // namespace proxddp
