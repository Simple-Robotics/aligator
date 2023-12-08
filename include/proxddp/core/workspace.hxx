/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/workspace.hpp"

namespace proxddp {

namespace {
template <typename T> long ncNonDyn(const ConstraintStackTpl<T> &cstrs) {
  return std::max(cstrs.totalDim() - cstrs.getDims()[0], 0L);
}
using proxnlp::isize;
} // namespace

using proxnlp::get_total_dim_helper;

template <typename Scalar>
WorkspaceTpl<Scalar>::WorkspaceTpl(const TrajOptProblemTpl<Scalar> &problem,
                                   LDLTChoice ldlt_choice)
    : Base(problem), stage_inner_crits(nsteps + 1),
      stage_dual_infeas(nsteps + 1) {

  prox_datas.reserve(nsteps + 1);

  Lxs_.reserve(nsteps + 1);
  Lus_.reserve(nsteps);

  prev_xs = trial_xs;
  prev_us = trial_us;
  kkt_mats_.reserve(nsteps + 1);
  kkt_rhs_.reserve(nsteps + 1);
  stage_prim_infeas.reserve(nsteps + 1);
  ldlts_.reserve(nsteps + 1);

  active_constraints.resize(nsteps + 1);
  lams_plus.resize(nsteps + 1);
  proj_jacobians.reserve(nsteps + 2);
  pd_step_.resize(nsteps + 1);
  dxs.reserve(nsteps + 1);
  dus.reserve(nsteps);
  dlams.reserve(nsteps + 1);
  dyn_slacks.reserve(nsteps);

  // initial condition
  if (nsteps > 0) {
    const int ndx1 = problem.stages_[0]->ndx1();
    const int nprim = ndx1;
    const int ndual = problem.init_condition_->nr;
    const int ntot = nprim + ndual;

    kkt_mats_.emplace_back(ntot, ntot);
    kkt_rhs_.emplace_back(ntot, ndx1 + 1);
    stage_prim_infeas.emplace_back(1);
    ldlts_.emplace_back(proxnlp::allocate_ldlt_from_sizes<Scalar>(
        {ndx1}, {ndual}, ldlt_choice));

    lams_plus[0] = VectorXs::Zero(ndual);
    proj_jacobians.emplace_back(ndual, ndx1);
    active_constraints[0] = VecBool::Zero(ndual);
    pd_step_[0] = VectorXs::Zero(ntot);
    dxs.emplace_back(pd_step_[0].head(ndx1));
    dlams.emplace_back(pd_step_[0].tail(ndual));
  } else {
    PROXDDP_WARNING("[Workspace]",
                    "Initialized a workspace for an empty problem (no nodes).");
    this->m_isInitialized = false;
    return;
  }

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];
    const int ndx1 = stage.ndx1();
    const int nu = stage.nu();
    const int ndx2 = stage.ndx2();
    const int nprim = stage.numPrimal();
    const int ndual = stage.numDual();
    // total matrix system dim
    const int ntot = nprim + ndual;
    const std::size_t ncb = stage.numConstraints();

    Lxs_.emplace_back(ndx1);
    Lus_.emplace_back(nu);

    value_params.emplace_back(ndx1);
    q_params.emplace_back(ndx1, nu, ndx2);

    kkt_mats_.emplace_back(ntot, ntot);
    kkt_rhs_.emplace_back(ntot, ndx1 + 1);
    ldlts_.emplace_back(proxnlp::allocate_ldlt_from_sizes<Scalar>(
        {nu, ndx2}, stage.constraints_.getDims(), ldlt_choice));
    stage_prim_infeas.emplace_back(ncb);

    lams_plus[i + 1] = VectorXs::Zero(ndual);
    proj_jacobians.emplace_back(ndual, ndx1 + nprim);
    active_constraints[i + 1] = VecBool::Zero(ndual);
    pd_step_[i + 1] = VectorXs::Zero(ntot);
    dus.emplace_back(pd_step_[i + 1].head(nu));
    dxs.emplace_back(pd_step_[i + 1].segment(nu, ndx2));
    dlams.emplace_back(pd_step_[i + 1].tail(ndual));
    dyn_slacks.push_back(dlams[i + 1].head(ndx2));
  }

  {
    const int ndx2 = problem.stages_.back()->ndx2();
    Lxs_.emplace_back(ndx2);
    value_params.emplace_back(ndx2);
  }

  // terminal node: always allocate data, even with dim 0
  if (!problem.term_cstrs_.empty()) {
    const int ndx1 = problem.stages_.back()->ndx2();
    const long ndual = problem.term_cstrs_.totalDim();
    stage_prim_infeas.emplace_back(1);
    lams_plus.push_back(VectorXs::Zero(ndual));
    proj_jacobians.emplace_back(ndual, ndx1);
    active_constraints.push_back(VecBool::Zero(ndual));
    pd_step_.push_back(VectorXs::Zero(ndual));
    dlams.push_back(pd_step_.back().tail(ndual));
  }

  math::setZero(Lxs_);
  math::setZero(Lus_);

  math::setZero(lams_plus);
  lams_pdal = lams_plus;
  trial_lams = lams_plus;
  prev_lams = lams_plus;
  Lds_ = prev_lams;
  shifted_constraints = lams_plus;

  math::setZero(kkt_mats_);
  math::setZero(kkt_rhs_);
  math::setZero(proj_jacobians);
  kkt_resdls_ = kkt_rhs_;

  stage_inner_crits.setZero();
  stage_dual_infeas.setZero();

  assert(value_params.size() == nsteps + 1);
  assert(dxs.size() == nsteps + 1);
  assert(dus.size() == nsteps);
}

template <typename Scalar> void WorkspaceTpl<Scalar>::cycleLeft() {
  Base::cycleLeft();

  rotate_vec_left(prox_datas);

  rotate_vec_left(cstr_scalers);
  rotate_vec_left(Lxs_);
  rotate_vec_left(Lus_);
  rotate_vec_left(Lds_);

  // number of "tail" multipliers that shouldn't be in the cycle
  long n_tail = 1;
  if (lams_plus.size() < (nsteps + 2)) {
    n_tail = 0;
  }

  rotate_vec_left(trial_lams, 1, n_tail);
  rotate_vec_left(lams_plus, 1, n_tail);
  rotate_vec_left(lams_pdal, 1, n_tail);
  rotate_vec_left(shifted_constraints, 1, n_tail);
  rotate_vec_left(proj_jacobians, 1, n_tail);
  rotate_vec_left(active_constraints, 1, n_tail);

  rotate_vec_left(pd_step_, 1, n_tail);
  rotate_vec_left(dxs);
  rotate_vec_left(dus);
  rotate_vec_left(dlams, 1, n_tail);

  rotate_vec_left(kkt_mats_, 1);
  rotate_vec_left(kkt_rhs_, 1);
  rotate_vec_left(kkt_resdls_, 1);
  // rotate_vec_left(ldlts_, 1);
  // std::rotate(ldlts_.begin(), ldlts_.begin() + 2, ldlts_.end());

  rotate_vec_left(prev_xs);
  rotate_vec_left(prev_us);
  rotate_vec_left(prev_lams, 1, n_tail);

  rotate_vec_left(stage_prim_infeas, 1, n_tail);
}

template <typename Scalar>
void WorkspaceTpl<Scalar>::configureScalers(
    const TrajOptProblemTpl<Scalar> &problem, const Scalar &mu) {
  cstr_scalers.reserve(nsteps + 1);

  for (std::size_t t = 0; t < nsteps; t++) {
    const StageModel &stage = *problem.stages_[t];
    cstr_scalers.emplace_back(stage.constraints_, mu);
    cstr_scalers[t].applyDefaultStrategy();
  }

  const ConstraintStackTpl<Scalar> &term_stack = problem.term_cstrs_;
  if (!term_stack.empty()) {
    cstr_scalers.emplace_back(term_stack, mu);
  }
}

template <typename Scalar>
std::ostream &operator<<(std::ostream &oss, const WorkspaceTpl<Scalar> &self) {
  oss << "Workspace {" << fmt::format("\n  nsteps:         {:d}", self.nsteps)
      << fmt::format("\n  n_multipliers:  {:d}", self.lams_pdal.size());
  oss << "\n}";
  return oss;
}

} // namespace proxddp
