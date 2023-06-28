/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/workspace.hpp"

namespace proxddp {

inline isize get_total_dim_helper(const std::vector<isize> &nprims,
                                  const std::vector<isize> &nduals) {
  return std::accumulate(nprims.begin(), nprims.end(), 0) +
         std::accumulate(nduals.begin(), nduals.end(), 0);
}

template <typename Scalar> struct custom_block_ldlt_allocator {
  using BlockLDLT = proxnlp::linalg::BlockLDLT<Scalar>;

  static BlockLDLT *create(const std::vector<isize> &nprims,
                           const std::vector<isize> &nduals,
                           bool primal_is_block_diagonal,
                           bool explicit_dynamics_block = true) {
    using proxnlp::linalg::BlockKind;
    using proxnlp::linalg::SymbolicBlockMatrix;

    SymbolicBlockMatrix structure =
        proxnlp::create_default_block_structure(nprims, nduals);

    if (primal_is_block_diagonal) {

      for (uint i = 0; i < nprims.size(); ++i) {
        for (uint j = 0; j < nprims.size(); ++j) {
          if (i != j) {
            structure(i, j) = BlockKind::Zero;
            structure(j, i) = BlockKind::Zero;
          }
        }
      }
    }
    if (explicit_dynamics_block && structure.nsegments() >= 3) {
      structure(2, 1) = BlockKind::Diag;
      structure(1, 2) = BlockKind::Diag;
    }
    isize size = get_total_dim_helper(nprims, nduals);
    return new BlockLDLT(size, structure);
  }
};

template <typename Scalar>
unique_ptr<ldlt_base<Scalar>>
allocate_ldlt_algorithm(const std::vector<isize> &nprims,
                        const std::vector<isize> &nduals, LDLTChoice choice) {
  using proxnlp::linalg::BlockLDLT;
  using proxnlp::linalg::DenseLDLT;
  using proxnlp::linalg::EigenLDLTWrapper;
  using proxnlp::linalg::SymbolicBlockMatrix;
  using ldlt_ptr = unique_ptr<ldlt_base<Scalar>>;

  isize size = get_total_dim_helper(nprims, nduals);

  switch (choice) {
  case LDLTChoice::DENSE:
    return ldlt_ptr(new DenseLDLT<Scalar>(size));
  case LDLTChoice::EIGEN:
    return ldlt_ptr(new EigenLDLTWrapper<Scalar>(size));
  case LDLTChoice::BLOCKED: {

    BlockLDLT<Scalar> *block_ptr =
        custom_block_ldlt_allocator<Scalar>::create(nprims, nduals, true);

    std::size_t nblocks = block_ptr->nblocks();
    std::vector<isize> perm((std::size_t)nblocks);
    std::iota(perm.begin(), perm.end(), 0);
    if (nprims.size() > 1) {
      std::rotate(perm.begin(), perm.begin() + 1, perm.end());
    }
#ifndef NDEBUG
    fmt::print("[block-ldlt] prior structure:\n");
    proxnlp::linalg::print_sparsity_pattern(block_ptr->structure());
    fmt::print("[block-ldlt] setting permutation = ({})\n",
               fmt::join(perm, ","));
#endif
    block_ptr->setBlockPermutation(perm.data());
    return ldlt_ptr(block_ptr);
  }
  case LDLTChoice::PROXSUITE: {
#ifdef PROXNLP_ENABLE_PROXSUITE_LDLT
    return ldlt_ptr(
        new proxnlp::linalg::ProxSuiteLDLTWrapper<Scalar>(size, nprims[0] + 2));
#else
    PROXDDP_RUNTIME_ERROR("ProxNLP was not compiled with ProxSuite support.");
#endif
  }
  default:
    return nullptr;
  }
}

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
    ldlts_.emplace_back(
        allocate_ldlt_algorithm<Scalar>({ndx1}, {ndual}, ldlt_choice));

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
    ldlts_.emplace_back(allocate_ldlt_algorithm<Scalar>(
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

  if (problem.term_cstrs_.size() > 0) {
    const int ndx1 = problem.stages_.back()->ndx2();
    const int nprim = ndx1;
    const long ndual = problem.term_cstrs_.totalDim();
    const long ntot = nprim + ndual;
    kkt_mats_.emplace_back(ntot, ntot);
    kkt_rhs_.emplace_back(ntot, ndx1 + 1);
    stage_prim_infeas.emplace_back(1);
    ldlts_.emplace_back(
        allocate_ldlt_algorithm<Scalar>({nprim}, {ndual}, ldlt_choice));

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
  lams_prev = lams_plus;
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

  rotate_vec_left(trial_lams, 1);
  rotate_vec_left(lams_plus, 1);
  rotate_vec_left(lams_pdal, 1);
  rotate_vec_left(shifted_constraints, 1);
  rotate_vec_left(proj_jacobians, 1);
  rotate_vec_left(active_constraints, 1);

  rotate_vec_left(pd_step_, 1);
  rotate_vec_left(dxs);
  rotate_vec_left(dus);
  rotate_vec_left(dlams);

  rotate_vec_left(kkt_mats_, 1);
  rotate_vec_left(kkt_rhs_, 1);
  rotate_vec_left(kkt_resdls_, 1);
  // rotate_vec_left(ldlts_, 1);
  // std::rotate(ldlts_.begin(), ldlts_.begin() + 2, ldlts_.end());

  rotate_vec_left(prev_xs);
  rotate_vec_left(prev_us);
  rotate_vec_left(lams_prev);

  rotate_vec_left(stage_prim_infeas);
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
