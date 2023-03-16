#pragma once

#include "proxddp/fddp/workspace.hpp"

namespace proxddp {

template <typename Scalar>
WorkspaceFDDPTpl<Scalar>::WorkspaceFDDPTpl(
    const TrajOptProblemTpl<Scalar> &problem)
    : Base(problem) {
  const std::size_t nsteps = this->nsteps;

  this->dyn_slacks.resize(nsteps + 1);
  dxs.resize(nsteps + 1);
  dus.resize(nsteps);
  Quuks_.resize(nsteps);
  ftVxx_.resize(nsteps + 1);
  kkt_mat_bufs.resize(nsteps);
  kkt_rhs_bufs.resize(nsteps);
  llts_.reserve(nsteps);
  JtH_temp_.reserve(nsteps);

  if (nsteps > 0) {
    const int ndx = problem.stages_[0]->ndx1();
    this->dyn_slacks[0] = VectorXs::Zero(ndx);
    ftVxx_[0] = VectorXs::Zero(ndx);
  } else {
    PROXDDP_WARNING("[WorkspaceFDDP]",
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
    kkt_mat_bufs[i] = MatrixXs::Zero(nu, nu);
    kkt_rhs_bufs[i] = MatrixXs::Zero(nu, ndx + 1);
    llts_.emplace_back(nu);
    JtH_temp_.emplace_back(ndx + nu, ndx);
    JtH_temp_.back().setZero();
  }
  const StageModelTpl<Scalar> &sm = *problem.stages_.back();
  dxs[nsteps] = VectorXs::Zero(sm.ndx2());
  value_params.emplace_back(sm.ndx2());

  assert(llts_.size() == nsteps);
}

template <typename Scalar> void WorkspaceFDDPTpl<Scalar>::cycleLeft() {
  Base::cycleLeft();

  rotate_vec_left(dxs);
  rotate_vec_left(dus);
  rotate_vec_left(Quuks_);
  rotate_vec_left(ftVxx_, 1);
  rotate_vec_left(kkt_mat_bufs);
  rotate_vec_left(kkt_rhs_bufs);
  rotate_vec_left(llts_);
  rotate_vec_left(JtH_temp_);
}

} // namespace proxddp
