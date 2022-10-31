#pragma once

namespace proxddp {

template <typename Scalar>
WorkspaceFDDPTpl<Scalar>::WorkspaceFDDPTpl(
    const TrajOptProblemTpl<Scalar> &problem)
    : Base(problem) {

  value_params.reserve(nsteps + 1);
  q_params.reserve(nsteps);

  xnexts_.resize(nsteps + 1);
  feas_gaps_.resize(nsteps + 1);
  dxs.resize(nsteps + 1);
  dus.resize(nsteps);
  Quuks_.resize(nsteps);
  ftVxx_.resize(nsteps + 1);
  kkt_mat_bufs.resize(nsteps);
  kkt_rhs_bufs.resize(nsteps);
  llts_.reserve(nsteps);

  feas_gaps_[0] = VectorXs::Zero(problem.stages_[0]->ndx1());

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModelTpl<Scalar> &sm = *problem.stages_[i];
    const int ndx = sm.ndx1();
    const int nu = sm.nu();
    const int ndual = sm.numDual();

    value_params.emplace_back(ndx);
    q_params.emplace_back(ndx, nu, 0);

    xnexts_[i] = sm.xspace().neutral();
    feas_gaps_[i + 1] = VectorXs::Zero(sm.ndx2());
    dxs[i] = VectorXs::Zero(ndx);
    dus[i] = VectorXs::Zero(nu);

    Quuks_[i] = VectorXs::Zero(sm.nu());
    ftVxx_[i + 1] = VectorXs::Zero(sm.ndx2());
    kkt_mat_bufs[i] = MatrixXs::Zero(nu, nu);
    kkt_rhs_bufs[i] = MatrixXs::Zero(nu, ndx + 1);
    llts_.emplace_back(nu);
  }
  const StageModelTpl<Scalar> &sm = *problem.stages_.back();
  dxs[nsteps] = VectorXs::Zero(sm.ndx2());
  xnexts_[nsteps] = sm.xspace_next().neutral();
  value_params.emplace_back(sm.ndx2());

  assert(llts_.size() == nsteps);
}

} // namespace proxddp
