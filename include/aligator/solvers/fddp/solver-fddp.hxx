/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "./solver-fddp.hpp"
#include "./linesearch.hpp"
#include "aligator/core/stage-data.hpp"

#include <fmt/ranges.h>

namespace aligator {

/* SolverFDDPTpl<Scalar> */

template <typename Scalar>
SolverFDDPTpl<Scalar>::SolverFDDPTpl(const Scalar tol, VerboseLevel verbose,
                                     const Scalar reg_init,
                                     const std::size_t max_iters)
    : target_tol_(tol), reg_init(reg_init), verbose_(verbose),
      max_iters(max_iters), force_initial_condition_(false), num_threads_(1) {
  ls_params.alpha_min = pow(2., -9.);
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::setup(const Problem &problem) {
  problem.checkIntegrity();
  results_.~Results();
  workspace_.~Workspace();
  new (&results_) Results(problem);
  new (&workspace_) Workspace(problem);
  // check if there are any constraints other than dynamics and throw a warning
  std::vector<std::size_t> idx_where_constraints;
  for (std::size_t i = 0; i < problem.numSteps(); i++) {
    const StageModel &sm = *problem.stages_[i];
    if (!sm.constraints_.empty())
      idx_where_constraints.push_back(i);
  }
  if (idx_where_constraints.size() > 0) {
    ALIGATOR_FDDP_WARNING(
        fmt::format("problem stages [{}] have constraints, "
                    "which this solver cannot handle. "
                    "Please use a penalized cost formulation.\n",
                    fmt::join(idx_where_constraints, ", ")));
  }
  if (!problem.term_cstrs_.empty()) {
    ALIGATOR_FDDP_WARNING("problem has at least one terminal constraint, which "
                          "this solver cannot "
                          "handle.\n");
  }
}

template <typename Scalar>
Scalar SolverFDDPTpl<Scalar>::forwardPass(const Problem &problem,
                                          const Results &results,
                                          Workspace &workspace,
                                          const Scalar alpha) {
  ALIGATOR_NOMALLOC_BEGIN;
  const std::size_t nsteps = workspace.nsteps;
  std::vector<VectorXs> &xs_try = workspace.trial_xs;
  std::vector<VectorXs> &us_try = workspace.trial_us;
  const std::vector<VectorXs> &fs = workspace.dyn_slacks;
  ProblemData &prob_data = workspace.problem_data;

  {
    const auto &space = problem.stages_[0]->xspace_;
    workspace.dxs[0] = alpha * fs[0];
    space->integrate(results.xs[0], workspace.dxs[0], xs_try[0]);
  }
  Scalar traj_cost_ = 0.;

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &sm = *problem.stages_[i];
    StageData &sd = *prob_data.stage_data[i];

    auto kkt_ff = results.getFeedforward(i);
    auto kkt_fb = results.getFeedback(i);

    workspace.dus[i] = alpha * kkt_ff;
    workspace.dus[i].noalias() += kkt_fb * workspace.dxs[i];
    sm.uspace().integrate(results.us[i], workspace.dus[i], us_try[i]);

    ALIGATOR_NOMALLOC_END;
    sm.evaluate(xs_try[i], us_try[i], xs_try[i + 1], sd);
    ALIGATOR_NOMALLOC_BEGIN;

    const ExplicitDynamicsData &dd = stage_get_dynamics_data(sd);

    workspace.dxs[i + 1] = (alpha - 1.) * fs[i + 1]; // use as tmp variable
    sm.xspace_next_->integrate(dd.xnext_, workspace.dxs[i + 1], xs_try[i + 1]);
    const CostData &cd = *sd.cost_data;

    ALIGATOR_RAISE_IF_NAN_NAME(xs_try[i + 1], fmt::format("xs[{}]", i + 1));
    ALIGATOR_RAISE_IF_NAN_NAME(us_try[i], fmt::format("us[{}]", i));

    sm.xspace_->difference(results.xs[i + 1], xs_try[i + 1],
                           workspace.dxs[i + 1]);

    traj_cost_ += cd.value_;
  }
  CostData &cd_term = *prob_data.term_cost_data;

  ALIGATOR_NOMALLOC_END;
  problem.term_cost_->evaluate(xs_try.back(), us_try.back(), cd_term);
  ALIGATOR_NOMALLOC_BEGIN;

  traj_cost_ += cd_term.value_;
  const auto &space = internal::problem_last_state_space_helper(problem);
  space->difference(results.xs[nsteps], xs_try[nsteps], workspace.dxs[nsteps]);

  prob_data.cost_ = traj_cost_;
  ALIGATOR_NOMALLOC_END;
  return traj_cost_;
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::expectedImprovement(Workspace &workspace,
                                                Scalar &d1, Scalar &d2) const {
  ALIGATOR_NOMALLOC_BEGIN;
  Scalar &dv = workspace.dv_;
  dv = 0.;
  const std::size_t nsteps = workspace.nsteps;

  for (std::size_t i = 0; i <= nsteps; i++) {
    const VectorXs &ftVxx = workspace.ftVxx_[i];
    dv -= workspace.dxs[i].dot(ftVxx);
  }

  d1 = workspace.dg_ + dv;
  d2 = workspace.dq_ - 2 * dv;
  ALIGATOR_NOMALLOC_END;
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::updateExpectedImprovement(Workspace &workspace,
                                                      Results &results) const {
  ALIGATOR_NOMALLOC_BEGIN;
  Scalar &dg = workspace.dg_;
  Scalar &dq = workspace.dq_;
  dg = 0.; // cost directional derivative
  dq = 0.; // cost 2nd direct. derivative
  const std::vector<VectorXs> &fs = workspace.dyn_slacks;
  const std::size_t nsteps = workspace.nsteps;

  // in croco: feedback/feedforward sign is flipped
  for (std::size_t i = 0; i < nsteps; i++) {
    const QParams &qparam = workspace.q_params[i];
    auto kkt_ff = results.getFeedforward(i);
    dg += qparam.Qu.dot(kkt_ff);
    dq += kkt_ff.dot(workspace.Quuks_[i]);
  }

  for (std::size_t i = 0; i <= nsteps; i++) {
    const VParams &vpar = workspace.value_params[i];
    dg += vpar.Vx_.dot(fs[i]);
    const VectorXs &ftVxx = workspace.ftVxx_[i];
    dq -= ftVxx.dot(fs[i]);
  }
  ALIGATOR_NOMALLOC_END;
}

template <typename Scalar>
Scalar SolverFDDPTpl<Scalar>::computeInfeasibility(const Problem &problem) {
  ALIGATOR_NOMALLOC_BEGIN;
  const std::size_t nsteps = problem.numSteps();
  const ProblemData &pd = workspace_.problem_data;
  const auto &xs = results_.xs;
  std::vector<VectorXs> &fs = workspace_.dyn_slacks;

  const auto &space = problem.stages_[0]->xspace_;
  space->difference(xs[0], problem.getInitState(), fs[0]);

#pragma omp parallel for num_threads(num_threads_)
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &sm = *problem.stages_[i];
    const auto &sd = *pd.stage_data[i];
    const ExplicitDynamicsData &dd = stage_get_dynamics_data(sd);
    sm.xspace_->difference(xs[i + 1], dd.xnext_, fs[i + 1]);
  }
  Scalar res = math::infty_norm(fs);
  ALIGATOR_NOMALLOC_END;
  return res;
}

template <typename Scalar>
Scalar SolverFDDPTpl<Scalar>::computeCriterion(Workspace &workspace) {
  ALIGATOR_NOMALLOC_BEGIN;
  const std::size_t nsteps = workspace.nsteps;
  Scalar v = 0.;
  for (std::size_t i = 0; i < nsteps; i++) {
    Scalar s = math::infty_norm(workspace.q_params[i].Qu);
    v = std::max(v, s);
  }
  ALIGATOR_NOMALLOC_END;
  return v;
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::backwardPass(const Problem &problem,
                                         Workspace &workspace) const {
  ALIGATOR_NOMALLOC_BEGIN;

  const std::size_t nsteps = workspace.nsteps;
  const std::vector<VectorXs> &fs = workspace.dyn_slacks;

  ProblemData &prob_data = workspace.problem_data;
  {
    const CostData &term_cost_data = *prob_data.term_cost_data;
    VParams &vp = workspace.value_params[nsteps];
    vp.v_ = term_cost_data.value_;
    vp.Vx_ = term_cost_data.Lx_;
    vp.Vxx_ = term_cost_data.Lxx_;
    vp.Vxx_.diagonal().array() += preg_;
    VectorXs &ftVxx = workspace.ftVxx_[nsteps];
    ftVxx.noalias() = vp.Vxx_ * fs[nsteps];
    vp.Vx_ += ftVxx;
  }

  for (std::size_t i = nsteps; i-- > 0;) {
    const VParams &vnext = workspace.value_params[i + 1];
    QParams &qparam = workspace.q_params[i];

    StageModel sm = *problem.stages_[i];
    StageData &sd = *prob_data.stage_data[i];

    const int nu = sm.nu();
    const int ndx1 = sm.ndx1();
    assert(qparam.grad_.size() == ndx1 + nu);

    const CostData &cd = *sd.cost_data;
    const DynamicsDataTpl<Scalar> &dd = *sd.dynamics_data;

    /* Assemble Q-function */
    auto J_x_u = dd.jac_buffer_.leftCols(ndx1 + nu);

    qparam.q_ = cd.value_;
    qparam.grad_ = cd.grad_;
    qparam.grad_.noalias() += J_x_u.transpose() * vnext.Vx_;

    // TODO: implement second-order derivatives for the Q-function
    workspace.JtH_temp_[i].noalias() = J_x_u.transpose() * vnext.Vxx_;
    qparam.hess_ = cd.hess_;
    qparam.hess_.noalias() += workspace.JtH_temp_[i] * J_x_u;
    qparam.Quu.diagonal().array() += preg_;

    /* Compute gains */

    MatrixXs &kkt_rhs = workspace.kktRhs[i];
    auto kkt_ff = kkt_rhs.col(0);
    auto kkt_fb = kkt_rhs.rightCols(ndx1);

    kkt_ff = -qparam.Qu;
    kkt_fb = -qparam.Qxu.transpose();

    Eigen::LLT<MatrixXs> &llt = workspace.llts_[i];
    llt.compute(qparam.Quu);
    llt.solveInPlace(kkt_rhs);

    workspace.Quuks_[i].noalias() = qparam.Quu * kkt_ff;

    /* Compute value function */
    VParams &vp = workspace.value_params[i];
    vp.Vx_ = qparam.Qx;
    vp.Vx_.noalias() += kkt_fb.transpose() * qparam.Qu;
    vp.Vxx_ = qparam.Qxx;
    vp.Vxx_.noalias() += qparam.Qxu * kkt_fb;
    vp.Vxx_ = vp.Vxx_.template selfadjointView<Eigen::Lower>();
    vp.Vxx_.diagonal().array() += preg_;
    VectorXs &ftVxx = workspace.ftVxx_[i];
    ftVxx.noalias() = vp.Vxx_ * fs[i];
    vp.Vx_ += ftVxx;
  }

  ALIGATOR_NOMALLOC_END;
}

template <typename Scalar>
bool SolverFDDPTpl<Scalar>::run(const Problem &problem,
                                const std::vector<VectorXs> &xs_init,
                                const std::vector<VectorXs> &us_init) {
  preg_ = reg_init;

  if (!results_.isInitialized() || !workspace_.isInitialized()) {
    ALIGATOR_RUNTIME_ERROR(
        "Either results or workspace not allocated. Call setup() first!");
  }

  check_trajectory_and_assign(problem, xs_init, us_init, results_.xs,
                              results_.us);
  // optionally override xs[0]
  if (force_initial_condition_) {
    workspace_.trial_xs[0] = problem.getInitState();
  }
  results_.conv = false;

  logger.active = verbose_ > 0;
  logger.addColumn(BASIC_KEYS[0]);
  logger.addColumn(BASIC_KEYS[1]);
  logger.addColumn(BASIC_KEYS[3]);
  logger.addColumn(BASIC_KEYS[4]);
  logger.addColumn(BASIC_KEYS[5]);
  logger.addColumn(BASIC_KEYS[6]);
  logger.addColumn(BASIC_KEYS[7]);
  logger.addColumn(BASIC_KEYS[9]);
  logger.printHeadline();

  // in Crocoddyl, linesearch xs is primed to use problem x0

  const auto linesearch_fun = [&](const Scalar alpha) {
    return forwardPass(problem, results_, workspace_, alpha);
  };

  Scalar &d1_phi = workspace_.d1_;
  Scalar &d2_phi = workspace_.d2_;
  Scalar phi0;
  // linesearch model oracle
  const auto ls_model = [&](const Scalar alpha) {
    expectedImprovement(workspace_, d1_phi, d2_phi);
    return phi0 + alpha * (d1_phi + 0.5 * d2_phi * alpha);
  };

  std::size_t &iter = results_.num_iters;
  results_.traj_cost_ = problem.evaluate(results_.xs, results_.us,
                                         workspace_.problem_data, num_threads_);

  for (iter = 0; iter < max_iters; ++iter) {

    problem.computeDerivatives(results_.xs, results_.us,
                               workspace_.problem_data, num_threads_);
    results_.prim_infeas = computeInfeasibility(problem);
    ALIGATOR_RAISE_IF_NAN(results_.prim_infeas);

    backwardPass(problem, workspace_);
    results_.dual_infeas = computeCriterion(workspace_);
    ALIGATOR_RAISE_IF_NAN(results_.dual_infeas);

    Scalar stopping_criterion =
        std::max(results_.prim_infeas, results_.dual_infeas);
    if (stopping_criterion < target_tol_) {
      results_.conv = true;
      break;
    }

    acceptGains(workspace_, results_);

    phi0 = results_.traj_cost_;
    ALIGATOR_RAISE_IF_NAN(phi0);

    updateExpectedImprovement(workspace_, results_);

    Scalar alpha_opt, phi_new;
    std::tie(alpha_opt, phi_new) = fddp_goldstein_linesearch(
        linesearch_fun, ls_model, phi0, ls_params, th_grad_, d1_phi);

    results_.traj_cost_ = phi_new;
    ALIGATOR_RAISE_IF_NAN(alpha_opt);
    ALIGATOR_RAISE_IF_NAN(phi_new);

    logger.addEntry("iter", iter + 1);
    logger.addEntry("alpha", alpha_opt);
    logger.addEntry("prim_err", results_.prim_infeas);
    logger.addEntry("dual_err", results_.dual_infeas);
    logger.addEntry("preg", preg_);
    logger.addEntry("dphi0", d1_phi);
    logger.addEntry("merit", phi_new);
    logger.addEntry("ΔM", phi_new - phi0);

    results_.xs = workspace_.trial_xs;
    results_.us = workspace_.trial_us;
    if (std::abs(d1_phi) < th_grad_) {
      results_.conv = true;
      break;
    }

    if (alpha_opt > th_step_dec_) {
      decreaseRegularization();
    }
    if (alpha_opt <= th_step_inc_) {
      increaseRegularization();
      if (preg_ == reg_max_) {
        results_.conv = false;
        break;
      }
    }

    invokeCallbacks(workspace_, results_);
    logger.log();
  }

  if (iter < max_iters)
    logger.log();
  logger.finish(results_.conv);
  return results_.conv;
}

template <typename Scalar>
auto SolverFDDPTpl<Scalar>::stage_get_dynamics_data(
    const StageDataTpl<Scalar> &data) -> const ExplicitDynamicsData & {
  const DynamicsDataTpl<Scalar> &dd = *data.dynamics_data;
  return static_cast<const ExplicitDynamicsData &>(dd);
}

} // namespace aligator
