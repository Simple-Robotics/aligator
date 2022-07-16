#pragma once

namespace proxddp {

template <typename Scalar>
ResultsTpl<Scalar>::ResultsTpl(const TrajOptProblemTpl<Scalar> &problem) {

  const std::size_t nsteps = problem.numSteps();
  gains_.reserve(nsteps);
  xs_.reserve(nsteps + 1);
  us_.reserve(nsteps);
  lams_.reserve(nsteps + 1);
  co_state_.reserve(nsteps);
  int ndual;
  ndual = problem.init_state_error.nr;
  lams_.push_back(VectorXs::Ones(ndual));
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModelTpl<Scalar> &stage = *problem.stages_[i];
    const int nprim = stage.numPrimal();
    ndual = stage.numDual();
    gains_.push_back(MatrixXs::Zero(nprim + ndual, stage.ndx1() + 1));
    xs_.push_back(stage.xspace_->neutral());
    us_.push_back(stage.uspace_->neutral());
    lams_.push_back(VectorXs::Ones(ndual));
    co_state_.push_back(lams_[i + 1].head(stage.dyn_model().nr));
    if (i == nsteps - 1)
      xs_.push_back(stage.xspace_next_->neutral());
  }

  if (problem.term_constraint_) {
    const StageConstraintTpl<Scalar> &tc = *problem.term_constraint_;
    const int ndx = tc.func_->ndx1;
    ndual = tc.func_->nr;
    lams_.push_back(VectorXs::Zero(ndual));
    gains_.push_back(MatrixXs::Zero(ndual, ndx + 1));
  }
  assert(xs_.size() == nsteps + 1);
  assert(us_.size() == nsteps);
}
} // namespace proxddp
