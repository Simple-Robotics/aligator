#pragma once

#include <fmt/ostream.h>

namespace proxddp {

namespace math {

template <typename T> void setZero(std::vector<T> &mats) {
  for (std::size_t i = 0; i < mats.size(); i++) {
    mats[i].setZero();
  }
}

} // namespace math

template <typename Scalar>
WorkspaceTpl<Scalar>::WorkspaceTpl(const TrajOptProblemTpl<Scalar> &problem)
    : Base(problem), inner_criterion_by_stage(nsteps + 1),
      primal_infeas_by_stage(nsteps + 1), dual_infeas_by_stage(nsteps + 1) {

  value_params.reserve(nsteps + 1);
  q_params.reserve(nsteps);
  prox_datas.reserve(nsteps + 1);

  prev_xs = trial_xs;
  prev_us = trial_us;
  kkt_mat_buf_.reserve(nsteps + 1);
  kkt_rhs_buf_.reserve(nsteps + 1);
  ldlts_.reserve(nsteps + 1);

  lams_plus.resize(nsteps + 1);
  pd_step_.resize(nsteps + 1);
  dxs_.reserve(nsteps + 1);
  dus_.reserve(nsteps);
  dlams_.reserve(nsteps + 1);
  co_state_.reserve(nsteps);

  {
    const int ndx1 = problem.stages_[0]->ndx1();
    const int nprim = ndx1;
    const int ndual = problem.init_state_error.nr;
    const int ntot = nprim + ndual;

    kkt_mat_buf_.emplace_back(ntot, ntot);
    kkt_rhs_buf_.emplace_back(ntot, ndx1 + 1);
    ldlts_.emplace_back(kkt_mat_buf_[0]);

    lams_plus[0] = VectorXs::Zero(ndual);
    pd_step_[0] = VectorXs::Zero(ntot);
    dxs_.emplace_back(pd_step_[0].head(ndx1));
    dlams_.emplace_back(pd_step_[0].tail(ndual));
  }

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];
    const int ndx1 = stage.ndx1(), nu = stage.nu();
    const int ndx2 = stage.ndx2();
    const int nprim = stage.numPrimal();
    const int ndual = stage.numDual();
    const int ntot = nprim + ndual;

    value_params.emplace_back(ndx1);
    q_params.emplace_back(ndx1, nu, ndx2);

    kkt_mat_buf_.emplace_back(ntot, ntot);
    kkt_rhs_buf_.emplace_back(ntot, ndx1 + 1);
    ldlts_.emplace_back(kkt_mat_buf_[i + 1]);

    lams_plus[i + 1] = VectorXs::Zero(ndual);
    pd_step_[i + 1] = VectorXs::Zero(ntot);
    dus_.emplace_back(pd_step_[i + 1].head(nu));
    dxs_.emplace_back(pd_step_[i + 1].segment(nu, ndx2));
    dlams_.emplace_back(pd_step_[i + 1].tail(ndual));
    co_state_.push_back(dlams_[i + 1].head(ndx2));
  }

  {
    const int ndx2 = problem.stages_.back()->ndx2();
    value_params.emplace_back(ndx2);
  }

  value_params_prev = value_params;

  if (problem.term_constraint_) {
    const StageConstraintTpl<Scalar> &tc = *problem.term_constraint_;
    const int ndx1 = tc.func->ndx1;
    const int nprim = ndx1;
    const int ndual = tc.func->nr;
    const int ntot = nprim + ndual;
    const Eigen::Index nc = primal_infeas_by_stage.size();
    primal_infeas_by_stage.conservativeResize(nc + 1);
    kkt_mat_buf_.emplace_back(ntot, ntot);
    kkt_rhs_buf_.emplace_back(ntot, ndx1 + 1);
    ldlts_.emplace_back(kkt_mat_buf_[nsteps]);

    lams_plus.push_back(VectorXs::Zero(ndual));
    pd_step_.push_back(VectorXs::Zero(ndual));
    dlams_.push_back(pd_step_.back().tail(ndual));
  }

  lams_pdal = lams_plus;
  trial_lams = lams_plus;
  prev_lams = lams_plus;

  math::setZero(kkt_mat_buf_);
  math::setZero(kkt_rhs_buf_);

  inner_criterion_by_stage.setZero();
  primal_infeas_by_stage.setZero();
  dual_infeas_by_stage.setZero();

  assert(value_params.size() == nsteps + 1);
  assert(dxs_.size() == nsteps + 1);
  assert(dus_.size() == nsteps);
}

template <typename Scalar>
std::ostream &operator<<(std::ostream &oss, const WorkspaceTpl<Scalar> &self) {
  oss << "Workspace {";
  oss << fmt::format("\n  nsteps       :  {:d}", self.nsteps)
      << fmt::format("\n  n_multipliers:  {:d}", self.lams_pdal.size());
  oss << "\n}";
  return oss;
}

} // namespace proxddp
