#pragma once

#include <fmt/ostream.h>

namespace proxddp {

template <typename Scalar>
WorkspaceTpl<Scalar>::WorkspaceTpl(const TrajOptProblemTpl<Scalar> &problem)
    : Base(problem), inner_criterion_by_stage(nsteps + 1),
      primal_infeas_by_stage(nsteps), dual_infeas_by_stage(nsteps + 1) {

  inner_criterion_by_stage.setZero();
  primal_infeas_by_stage.setZero();
  dual_infeas_by_stage.setZero();

  value_params.reserve(nsteps + 1);
  q_params.reserve(nsteps);
  prox_datas.reserve(nsteps + 1);

  prev_xs_ = trial_xs_;
  prev_us_ = trial_us_;

  lams_plus_.reserve(nsteps + 1);
  lams_pdal_.reserve(nsteps + 1);

  trial_lams_.reserve(nsteps + 1);
  prev_lams_.reserve(nsteps + 1);

  int nprim, ndual, ndx1, nu, ndx2;
  int max_kkt_size = 0;
  ndx1 = problem.stages_[0]->ndx1();
  nprim = ndx1;
  ndual = problem.init_state_error.nr;
  int max_ndx = nprim + ndual;
  pd_step_.push_back(VectorXs::Zero(nprim + ndual));
  lams_plus_.push_back(VectorXs::Zero(ndual));
  lams_pdal_.push_back(lams_plus_[0]);
  trial_lams_.push_back(lams_plus_[0]);
  prev_lams_.push_back(lams_plus_[0]);
  dxs_.push_back(pd_step_[0].head(ndx1));
  dlams_.push_back(pd_step_[0].tail(ndual));

  std::size_t i = 0;
  for (i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];
    ndx1 = stage.ndx1(), nu = stage.nu();
    ndx2 = stage.ndx2();
    nprim = stage.numPrimal();
    ndual = stage.numDual();

    value_params.push_back(value_storage_t(ndx1));
    q_params.push_back(q_storage_t(ndx1, nu, ndx2));

    pd_step_.push_back(VectorXs::Zero(nprim + ndual));

    dxs_.push_back(pd_step_[i + 1].segment(nu, ndx2));
    dus_.push_back(pd_step_[i + 1].head(nu));
    dlams_.push_back(pd_step_[i + 1].tail(ndual));

    lams_plus_.push_back(VectorXs::Zero(ndual));
    lams_pdal_.push_back(lams_plus_.back());

    trial_lams_.push_back(VectorXs::Zero(ndual));

    prev_lams_.push_back(trial_lams_.back());

    /** terminal node **/
    if (i == nsteps - 1) {
      value_params.push_back(value_storage_t(ndx2));
    }

    max_kkt_size = std::max(max_kkt_size, nprim + ndual);
    max_ndx = std::max(max_ndx, ndx2);
  }

  if (problem.term_constraint_) {
    const StageConstraintTpl<Scalar> &tc = *problem.term_constraint_;
    nprim = tc.func_->ndx1;
    ndual = tc.func_->nr;
    max_kkt_size = std::max(max_kkt_size, ndual);
    lams_plus_.push_back(VectorXs::Zero(ndual));
    lams_pdal_.push_back(VectorXs::Zero(ndual));
    trial_lams_.push_back(VectorXs::Zero(ndual));
    prev_lams_.push_back(VectorXs::Zero(ndual));

    pd_step_.push_back(VectorXs::Zero(ndual));
    dlams_.push_back(pd_step_.back().tail(ndual));
  }

  kktMatrixFull_.resize(max_kkt_size, max_kkt_size);
  kktMatrixFull_.setZero();

  kktRhsFull_.resize(max_kkt_size, max_ndx + 1);
  kktRhsFull_.setZero();

  assert(value_params.size() == nsteps + 1);
  assert(dxs_.size() == nsteps + 1);
  assert(dus_.size() == nsteps);
}

template <typename Scalar>
std::ostream &operator<<(std::ostream &oss, const WorkspaceTpl<Scalar> &self) {
  oss << "Workspace {";
  oss << fmt::format("\n  num nodes      : {:d}", self.trial_us_.size())
      << fmt::format("\n  kkt buffer size: {:d}", self.kktMatrixFull_.rows());
  oss << "\n}";
  return oss;
}

} // namespace proxddp
