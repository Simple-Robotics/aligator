#pragma once

#include <fmt/ostream.h>

namespace proxddp {

template <typename Scalar>
WorkspaceBaseTpl<Scalar>::WorkspaceBaseTpl(
    const TrajOptProblemTpl<Scalar> &problem)
    : nsteps(problem.numSteps()), problem_data(problem) {
  trial_xs.resize(nsteps + 1);
  trial_us.resize(nsteps);
  xs_default_init(problem, trial_xs);
  us_default_init(problem, trial_us);
}

template <typename Scalar> void WorkspaceBaseTpl<Scalar>::cycle_left() {
  rotate_vec_left(problem_data.stage_data);

  rotate_vec_left(trial_xs);
  rotate_vec_left(trial_us);

  rotate_vec_left(dyn_slacks);

  rotate_vec_left(value_params);
  rotate_vec_left(q_params);
}

template <typename Scalar>
WorkspaceTpl<Scalar>::WorkspaceTpl(const TrajOptProblemTpl<Scalar> &problem)
    : Base(problem), trial_prob_data(problem), stage_inner_crits(nsteps + 1),
      stage_dual_infeas(nsteps + 1) {

  value_params.reserve(nsteps + 1);
  q_params.reserve(nsteps);
  prox_datas.reserve(nsteps + 1);

  prev_xs = trial_xs;
  prev_us = trial_us;
  kkt_mat_buf_.reserve(nsteps + 1);
  kkt_rhs_buf_.reserve(nsteps + 1);
  stage_prim_infeas.reserve(nsteps + 1);
  ldlts_.reserve(nsteps + 1);

  lams_plus.resize(nsteps + 1);
  pd_step_.resize(nsteps + 1);
  dxs.reserve(nsteps + 1);
  dus.reserve(nsteps);
  dlams.reserve(nsteps + 1);
  co_states_.reserve(nsteps);
  this->dyn_slacks.reserve(nsteps);

  // initial condition
  {
    const int ndx1 = problem.stages_[0]->ndx1();
    const int nprim = ndx1;
    const int ndual = problem.init_state_error.nr;
    const int ntot = nprim + ndual;

    kkt_mat_buf_.emplace_back(ntot, ntot);
    kkt_rhs_buf_.emplace_back(ntot, ndx1 + 1);
    stage_prim_infeas.emplace_back(1);
    ldlts_.emplace_back(kkt_mat_buf_[0]);

    lams_plus[0] = VectorXs::Zero(ndual);
    pd_step_[0] = VectorXs::Zero(ntot);
    dxs.emplace_back(pd_step_[0].head(ndx1));
    dlams.emplace_back(pd_step_[0].tail(ndual));
  }

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];
    const int ndx1 = stage.ndx1(), nu = stage.nu();
    const int ndx2 = stage.ndx2();
    const int nprim = stage.numPrimal();
    const int ndual = stage.numDual();
    const int ntot = nprim + ndual;
    const std::size_t ncb = stage.numConstraints();

    value_params.emplace_back(ndx1);
    q_params.emplace_back(ndx1, nu, ndx2);

    kkt_mat_buf_.emplace_back(ntot, ntot);
    kkt_rhs_buf_.emplace_back(ntot, ndx1 + 1);
    stage_prim_infeas.emplace_back(ncb);
    ldlts_.emplace_back(kkt_mat_buf_[i + 1]);

    lams_plus[i + 1] = VectorXs::Zero(ndual);
    pd_step_[i + 1] = VectorXs::Zero(ntot);
    dus.emplace_back(pd_step_[i + 1].head(nu));
    dxs.emplace_back(pd_step_[i + 1].segment(nu, ndx2));
    dlams.emplace_back(pd_step_[i + 1].tail(ndual));
    this->dyn_slacks.push_back(dlams[i + 1].head(ndx2));
  }

  {
    const int ndx2 = problem.stages_.back()->ndx2();
    value_params.emplace_back(ndx2);
  }

  if (problem.term_constraint_) {
    const StageConstraintTpl<Scalar> &tc = *problem.term_constraint_;
    const int ndx1 = tc.func->ndx1;
    const int nprim = ndx1;
    const int ndual = tc.func->nr;
    const int ntot = nprim + ndual;
    kkt_mat_buf_.emplace_back(ntot, ntot);
    kkt_rhs_buf_.emplace_back(ntot, ndx1 + 1);
    stage_prim_infeas.emplace_back(1);
    ldlts_.emplace_back(kkt_mat_buf_[nsteps]);

    lams_plus.push_back(VectorXs::Zero(ndual));
    pd_step_.push_back(VectorXs::Zero(ndual));
    dlams.push_back(pd_step_.back().tail(ndual));
  }

  lams_pdal = lams_plus;
  trial_lams = lams_plus;
  lams_prev = lams_plus;
  shifted_constraints = lams_plus;
  this->co_states_ = this->dyn_slacks;

  math::setZero(kkt_mat_buf_);
  math::setZero(kkt_rhs_buf_);
  kkt_resdls_ = kkt_rhs_buf_;

  stage_inner_crits.setZero();
  stage_dual_infeas.setZero();

  assert(value_params.size() == nsteps + 1);
  assert(dxs.size() == nsteps + 1);
  assert(dus.size() == nsteps);
}

template <typename Scalar> void WorkspaceTpl<Scalar>::cycle_left() {
  Base::cycle_left();

  rotate_vec_left(trial_prob_data.stage_data);

  rotate_vec_left(co_states_);
  rotate_vec_left(prox_datas);
  rotate_vec_left(lams_plus);
  rotate_vec_left(lams_pdal);
  rotate_vec_left(shifted_constraints);
  rotate_vec_left(pd_step_);
  rotate_vec_left(dxs);
  rotate_vec_left(dus);
  rotate_vec_left(dlams);

  rotate_vec_left(kkt_mat_buf_);
  rotate_vec_left(kkt_rhs_buf_);
  rotate_vec_left(kkt_resdls_);
  rotate_vec_left(ldlts_);

  rotate_vec_left(prev_xs);
  rotate_vec_left(prev_us);
  rotate_vec_left(lams_prev);
}

template <typename Scalar>
std::ostream &operator<<(std::ostream &oss, const WorkspaceTpl<Scalar> &self) {
  oss << "Workspace {" << fmt::format("\n  nsteps:         {:d}", self.nsteps)
      << fmt::format("\n  n_multipliers:  {:d}", self.lams_pdal.size());
  oss << "\n}";
  return oss;
}

} // namespace proxddp
