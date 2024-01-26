#pragma once

#include "./workspace.hpp"

namespace aligator {

template <typename Scalar>
WorkspaceFDDPTpl<Scalar>::WorkspaceFDDPTpl(
    const TrajOptProblemTpl<Scalar> &problem)
    : Base(problem) {
  const std::size_t nsteps = this->nsteps;
  problem.checkIntegrity();

  this->dyn_slacks.resize(nsteps + 1);
  dxs.resize(nsteps + 1);
  dus.resize(nsteps);
  Quuks_.resize(nsteps);
  ftVxx_.resize(nsteps + 1);
  kktRhs.resize(nsteps);
  llts_.reserve(nsteps);
  JtH_temp_.reserve(nsteps);
  value_params.reserve(nsteps + 1);
  q_params.reserve(nsteps);

  if (nsteps > 0) {
    const int ndx = problem.stages_[0]->ndx1();
    this->dyn_slacks[0] = VectorXs::Zero(ndx);
    ftVxx_[0] = VectorXs::Zero(ndx);
  } else {
    ALIGATOR_WARNING(
        "[WorkspaceFDDP]",
        "Initialized a workspace for an empty problem (no nodes).");
    this->m_isInitialized = false;
    return;
  }

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModelTpl<Scalar> &sm = *problem.stages_[i];
    const int ndx = sm.ndx1();
    const int nu = sm.nu();

    value_params.emplace_back(ndx);
    q_params.emplace_back(ndx, nu, 0);

    this->dyn_slacks[i + 1] = VectorXs::Zero(sm.ndx2());
    dxs[i] = VectorXs::Zero(ndx);
    dus[i] = VectorXs::Zero(nu);

    Quuks_[i] = VectorXs::Zero(nu);
    ftVxx_[i + 1] = VectorXs::Zero(sm.ndx2());
    kktRhs[i] = MatrixXs::Zero(nu, ndx + 1);
    llts_.emplace_back(nu);
    JtH_temp_.emplace_back(ndx + nu, ndx);
    JtH_temp_.back().setZero();
  }
  const int ndx = internal::problem_last_ndx_helper(problem);
  dxs[nsteps].setZero(ndx);
  value_params.emplace_back(ndx);

  assert(llts_.size() == nsteps);
}

template <typename Scalar> void WorkspaceFDDPTpl<Scalar>::cycleLeft() {
  Base::cycleLeft();

  rotate_vec_left(dxs);
  rotate_vec_left(dus);
  rotate_vec_left(Quuks_);
  rotate_vec_left(ftVxx_, 1);
  rotate_vec_left(kktRhs);
  rotate_vec_left(llts_);
  rotate_vec_left(JtH_temp_);

  rotate_vec_left(value_params);
  rotate_vec_left(q_params);
}

} // namespace aligator
