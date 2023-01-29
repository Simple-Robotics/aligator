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
                           bool primal_is_block_diagonal) {
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
    block_ptr->setPermutation(perm.data());
    return ldlt_ptr(block_ptr);
  }
  default:
    return nullptr;
  }
}

template <typename Scalar>
WorkspaceTpl<Scalar>::WorkspaceTpl(const TrajOptProblemTpl<Scalar> &problem,
                                   LDLTChoice ldlt_choice)
    : Base(problem), stage_inner_crits(this->nsteps + 1),
      stage_dual_infeas(this->nsteps + 1) {
  const std::size_t nsteps = this->nsteps;

  value_params.reserve(nsteps + 1);
  q_params.reserve(nsteps);
  prox_datas.reserve(nsteps + 1);

  prev_xs = this->trial_xs;
  prev_us = this->trial_us;
  kkt_mats_.reserve(nsteps + 1);
  kkt_rhs_.reserve(nsteps + 1);
  stage_prim_infeas.reserve(nsteps + 1);
  ldlts_.reserve(nsteps + 1);

  lams_plus.resize(nsteps + 1);
  pd_step_.resize(nsteps + 1);
  dxs.reserve(nsteps + 1);
  dus.reserve(nsteps);
  dlams.reserve(nsteps + 1);
  this->dyn_slacks.reserve(nsteps);

  // initial condition
  {
    const int ndx1 = problem.stages_[0]->ndx1();
    const int nprim = ndx1;
    const int ndual = problem.init_state_error_.nr;
    const int ntot = nprim + ndual;

    kkt_mats_.emplace_back(ntot, ntot);
    kkt_rhs_.emplace_back(ntot, ndx1 + 1);
    stage_prim_infeas.emplace_back(1);
    ldlts_.emplace_back(
        allocate_ldlt_algorithm<Scalar>({ndx1}, {ndual}, ldlt_choice));

    lams_plus[0] = VectorXs::Zero(ndual);
    pd_step_[0] = VectorXs::Zero(ntot);
    dxs.emplace_back(pd_step_[0].head(ndx1));
    dlams.emplace_back(pd_step_[0].tail(ndual));
  }

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];
    const int ndx1 = stage.ndx1();
    const int nu = stage.nu();
    const int ndx2 = stage.ndx2();
    const int nprim = stage.numPrimal();
    const int ndual = stage.numDual();
    const int ntot = nprim + ndual;
    const std::size_t ncb = stage.numConstraints();

    value_params.emplace_back(ndx1);
    q_params.emplace_back(ndx1, nu, ndx2);

    kkt_mats_.emplace_back(ntot, ntot);
    kkt_rhs_.emplace_back(ntot, ndx1 + 1);
    ldlts_.emplace_back(allocate_ldlt_algorithm<Scalar>(
        {nu, ndx2}, stage.constraints_.getDims(), ldlt_choice));
    stage_prim_infeas.emplace_back(ncb);

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
    kkt_mats_.emplace_back(ntot, ntot);
    kkt_rhs_.emplace_back(ntot, ndx1 + 1);
    stage_prim_infeas.emplace_back(1);
    ldlts_.emplace_back(
        allocate_ldlt_algorithm<Scalar>({nprim}, {ndual}, ldlt_choice));

    lams_plus.push_back(VectorXs::Zero(ndual));
    pd_step_.push_back(VectorXs::Zero(ndual));
    dlams.push_back(pd_step_.back().tail(ndual));
  }

  lams_pdal = lams_plus;
  trial_lams = lams_plus;
  lams_prev = lams_plus;
  shifted_constraints = lams_plus;

  math::setZero(kkt_mats_);
  math::setZero(kkt_rhs_);
  kkt_resdls_ = kkt_rhs_;

  stage_inner_crits.setZero();
  stage_dual_infeas.setZero();

  assert(value_params.size() == nsteps + 1);
  assert(dxs.size() == nsteps + 1);
  assert(dus.size() == nsteps);
}

template <typename Scalar> void WorkspaceTpl<Scalar>::cycle_left() {
  Base::cycle_left();

  rotate_vec_left(prox_datas);
  rotate_vec_left(lams_plus, 1);
  rotate_vec_left(lams_pdal, 1);
  rotate_vec_left(shifted_constraints, 1);
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
}

template <typename Scalar>
void WorkspaceTpl<Scalar>::cycle_append(const shared_ptr<StageModel> &stage) {
  auto sd = stage->createData();
  problem_data.stage_data.push_back(sd);
  this->cycle_left();
  problem_data.stage_data.pop_back();
}

template <typename Scalar>
std::ostream &operator<<(std::ostream &oss, const WorkspaceTpl<Scalar> &self) {
  oss << "Workspace {" << fmt::format("\n  nsteps:         {:d}", self.nsteps)
      << fmt::format("\n  n_multipliers:  {:d}", self.lams_pdal.size());
  oss << "\n}";
  return oss;
}

} // namespace proxddp
