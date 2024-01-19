/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "./workspace.hpp"
#include "aligator/gar/utils.hpp"

namespace aligator {

template <typename Scalar>
WorkspaceTpl<Scalar>::WorkspaceTpl(const TrajOptProblemTpl<Scalar> &problem)
    : Base(problem), stage_inner_crits(nsteps + 1),
      stage_prim_infeas(nsteps + 1), state_dual_infeas(nsteps + 1),
      control_dual_infeas(nsteps + 1) {

  if (nsteps == 0) {
    ALIGATOR_WARNING(
        "[Workspace]",
        "Initialized a workspace for an empty problem (no nodes).");
    this->m_isInitialized = false;
    return;
  }

  std::tie(trial_xs, trial_us, trial_vs, trial_lams) =
      problemInitializeSolution(problem);
  std::tie(prev_xs, prev_us, prev_vs, prev_lams) = {trial_xs, trial_us,
                                                    trial_vs, trial_lams};

  vs_plus = vs_pdal = trial_vs;
  lams_plus = lams_pdal = trial_lams;

  dyn_slacks = trial_lams; // same dimensions

  constraintProductOperators.resize(nsteps + 1); // includes terminal
  shifted_constraints = prev_vs;

  active_constraints.resize(nsteps + 1);
  proj_jacobians.resize(nsteps + 1);

  std::vector<KnotType> knots;
  {
    for (size_t i = 0; i < nsteps; i++) {
      const StageModel &stage = *problem.stages_[i];
      knots.emplace_back(stage.ndx1(), stage.nu(), stage.nc());
    }

    knots.emplace_back(problem.stages_.back()->ndx2(), 0,
                       problem.term_cstrs_.totalDim());
  }

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];
    const int ndx1 = stage.ndx1();
    const int nu = stage.nu();
    const int ncstr = stage.nc();
    // total matrix system dim
    const std::size_t ncb = stage.numConstraints();

    constraintProductOperators[i] = getConstraintProductSet(stage.constraints_);
    stage_prim_infeas[i].setZero(ncb);
    proj_jacobians[i].setZero(ncstr, ndx1 + nu);
    active_constraints[i].setZero(ncstr);
  }

  // terminal node: always allocate data, even with dim 0
  if (!problem.term_cstrs_.empty()) {
    const int ndx1 = problem.stages_.back()->ndx2();
    const long nc = problem.term_cstrs_.totalDim();
    stage_prim_infeas[nsteps].setZero(problem.term_cstrs_.size());
    proj_jacobians[nsteps].setZero(nc, ndx1);
    active_constraints[nsteps].setZero(nc);
  }

  // initial condition
  long nc0 = (long)problem.init_condition_->nr;
  lqr_problem = gar::LQRProblemTpl<Scalar>(knots, nc0);
  std::tie(dxs, dus, dvs, dlams) =
      gar::lqrInitializeSolution(lqr_problem); // lqr subproblem variables
  Lxs_ = dxs;
  Lus_ = dus;
  Lvs_ = dvs;
  Lds_ = dlams;

  stage_inner_crits.setZero();
  state_dual_infeas.setZero();
  control_dual_infeas.setZero();
}

template <typename Scalar> void WorkspaceTpl<Scalar>::cycleLeft() {
  Base::cycleLeft();

  rotate_vec_left(cstr_scalers);
  rotate_vec_left(Lxs_);
  rotate_vec_left(Lus_);
  rotate_vec_left(Lds_);
  rotate_vec_left(Lvs_);

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

  rotate_vec_left(dxs);
  rotate_vec_left(dus);
  rotate_vec_left(dvs);
  rotate_vec_left(dlams, 1, n_tail);

  rotate_vec_left(prev_xs);
  rotate_vec_left(prev_us);
  rotate_vec_left(prev_lams, 1, n_tail);

  rotate_vec_left(stage_prim_infeas, 1, n_tail);
}

template <typename Scalar>
std::ostream &operator<<(std::ostream &oss, const WorkspaceTpl<Scalar> &self) {
  oss << "Workspace {" << fmt::format("\n  nsteps:         {:d}", self.nsteps)
      << fmt::format("\n  n_multipliers:  {:d}", self.lams_pdal.size());
  oss << "\n}";
  return oss;
}

} // namespace aligator
