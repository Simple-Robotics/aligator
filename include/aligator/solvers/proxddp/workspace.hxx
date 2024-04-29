/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "./workspace.hpp"
#include "aligator/gar/utils.hpp"

namespace aligator {

template <typename Scalar>
WorkspaceTpl<Scalar>::WorkspaceTpl(const TrajOptProblemTpl<Scalar> &problem)
    : Base(problem), stage_inner_crits(nsteps + 1),
      stage_cstr_violations(nsteps + 1), stage_infeasibilities(nsteps + 1),
      state_dual_infeas(nsteps + 1), control_dual_infeas(nsteps + 1) {

  problem.checkIntegrity();

  std::tie(trial_xs, trial_us, trial_vs, trial_lams) =
      problemInitializeSolution(problem);
  std::tie(prev_xs, prev_us, prev_vs, prev_lams) = {trial_xs, trial_us,
                                                    trial_vs, trial_lams};

  vs_plus = vs_pdal = trial_vs;
  lams_plus = lams_pdal = trial_lams;

  dyn_slacks = trial_lams; // same dimensions
  stage_cstr_violations.setZero();
  stage_infeasibilities = trial_vs;

  constraintProductOperators.reserve(nsteps + 1); // includes terminal
  shifted_constraints = prev_vs;

  active_constraints.resize(nsteps + 1);
  constraintProjJacobians.resize(nsteps + 1);

  using LQRProblemType = gar::LQRProblemTpl<Scalar>;
  typename LQRProblemType::KnotVector knots;
  {
    for (size_t i = 0; i < nsteps; i++) {
      const StageModel &stage = *problem.stages_[i];
      knots.emplace_back(stage.ndx1(), stage.nu(), stage.nc());
    }

    knots.emplace_back(internal::problem_last_ndx_helper(problem), 0,
                       problem.term_cstrs_.totalDim(), 0);
  }

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];
    const ConstraintStackTpl<Scalar> &stack = stage.constraints_;
    const int ndx1 = stage.ndx1();
    const int nu = stage.nu();
    const int ncstr = stage.nc();

    constraintProductOperators.emplace_back(
        getConstraintProductSet(stage.constraints_));
    constraintProjJacobians[i] = BlkJacobianType(stack.dims(), {ndx1, nu});
    active_constraints[i].setZero(ncstr);
  }

  // terminal node: always allocate, no check for nonempty constaint stack
  {
    const ConstraintStackTpl<Scalar> &stack = problem.term_cstrs_;
    const int ndx1 = internal::problem_last_ndx_helper(problem);
    constraintProductOperators.emplace_back(
        getConstraintProductSet(problem.term_cstrs_));
    constraintProjJacobians[nsteps] = BlkJacobianType(stack.dims(), {ndx1, 0});
    active_constraints[nsteps].setZero(stack.totalDim());
  }

  // initial condition
  long nc0 = (long)problem.init_condition_->nr;
  lqr_problem = LQRProblemType(knots, nc0);
  std::tie(dxs, dus, dvs, dlams) =
      gar::lqrInitializeSolution(lqr_problem); // lqr subproblem variables
  Lxs = dxs;
  Lus = dus;
  Lvs = dvs;
  Lds = dlams;
  constraintLxCorr = Lxs;
  constraintLuCorr = Lus;

  stage_inner_crits.setZero();
  state_dual_infeas.setZero();
  control_dual_infeas.setZero();
}

template <typename Scalar> void WorkspaceTpl<Scalar>::cycleLeft() {
  Base::cycleLeft();

  rotate_vec_left(cstr_scalers);
  rotate_vec_left(Lxs);
  rotate_vec_left(Lus);
  rotate_vec_left(Lds);
  rotate_vec_left(Lvs);

  rotate_vec_left(trial_lams, 1);
  rotate_vec_left(lams_plus, 1);
  rotate_vec_left(lams_pdal, 1);
  rotate_vec_left(shifted_constraints, 0, 1);
  rotate_vec_left(constraintProjJacobians, 0, 1);
  rotate_vec_left(active_constraints, 0, 1);

  rotate_vec_left(dxs);
  rotate_vec_left(dus);
  rotate_vec_left(dvs, 0, 1);
  rotate_vec_left(dlams, 1);

  rotate_vec_left(prev_xs);
  rotate_vec_left(prev_us);
  rotate_vec_left(prev_vs, 0, 1);
  rotate_vec_left(prev_lams, 1);

  rotate_vec_left(stage_infeasibilities, 0, 1);
}

template <typename Scalar>
std::ostream &operator<<(std::ostream &oss, const WorkspaceTpl<Scalar> &self) {
  oss << "Workspace {" << fmt::format("\n  nsteps:         {:d}", self.nsteps)
      << fmt::format("\n  n_multipliers:  {:d}", self.lams_pdal.size());
  oss << "\n}";
  return oss;
}

} // namespace aligator
