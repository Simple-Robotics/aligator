/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
/// @brief Implementation file, to be included when necessary.
#pragma once

#include "./workspace.hpp"
#include "aligator/gar/lqr-problem.hpp"
#include "aligator/gar/utils.hpp"

namespace aligator {

template <typename Scalar>
WorkspaceTpl<Scalar>::WorkspaceTpl(const TrajOptProblemTpl<Scalar> &problem,
                                   const allocator_type &alloc)
    : Base(problem)
    , lqr_problem(alloc)
    , stage_inner_crits(nsteps + 1)
    , stage_cstr_violations(nsteps + 1)
    , stage_infeasibilities(nsteps + 1)
    , state_dual_infeas(nsteps + 1)
    , control_dual_infeas(nsteps + 1) {

  if (!problem.checkIntegrity())
    ALIGATOR_RUNTIME_ERROR("Problem failed integrity check.");

  problem.initializeSolution(trial_xs, trial_us, trial_vs, trial_lams);
  prev_xs = trial_xs;
  prev_us = trial_us;
  prev_vs = trial_vs;

  vs_plus = vs_pdal = trial_vs;
  lams_plus = trial_lams;

  dyn_slacks = trial_lams; // same dimensions
  stage_cstr_violations.setZero();
  stage_infeasibilities = trial_vs;

  cstr_product_sets.reserve(nsteps + 1); // includes terminal
  shifted_constraints = prev_vs;

  active_constraints.resize(nsteps + 1);
  cstr_proj_jacs.resize(nsteps + 1);

  auto &knots = lqr_problem.stages;

  for (size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];
    knots.emplace_back(stage.ndx1(), stage.nu(), stage.nc());
  }

  knots.emplace_back(internal::problem_last_ndx_helper(problem), 0,
                     problem.term_cstrs_.totalDim(), 0);

  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModel &stage = *problem.stages_[i];
    const ConstraintStackTpl<Scalar> &stack = stage.constraints_;
    const int ndx1 = stage.ndx1();
    const int nu = stage.nu();
    const int ncstr = stage.nc();

    cstr_product_sets.emplace_back(getConstraintProductSet(stage.constraints_));
    cstr_proj_jacs[i] = BlkJacobianType(stack.dims(), {ndx1, nu});
    active_constraints[i].setZero(ncstr);
  }

  // terminal node: always allocate, no check for nonempty constaint stack
  {
    const ConstraintStackTpl<Scalar> &stack = problem.term_cstrs_;
    const int ndx1 = internal::problem_last_ndx_helper(problem);
    cstr_product_sets.emplace_back(
        getConstraintProductSet(problem.term_cstrs_));
    cstr_proj_jacs[nsteps] = BlkJacobianType(stack.dims(), {ndx1, 0});
    active_constraints[nsteps].setZero(stack.totalDim());
  }

  // initial condition
  long nc0 = (long)problem.init_constraint_->nr;
  lqr_problem.G0.resize(nc0, nc0);
  lqr_problem.g0.resize(nc0);
  std::tie(dxs, dus, dvs, dlams) =
      gar::lqrInitializeSolution(lqr_problem); // lqr subproblem variables
  Lxs = dxs;
  Lus = dus;
  Lvs = dvs;
  cstr_lx_corr = Lxs;
  cstr_lu_corr = Lus;

  stage_inner_crits.setZero();
  state_dual_infeas.setZero();
  control_dual_infeas.setZero();
}

template <typename Scalar>
void WorkspaceTpl<Scalar>::cycleAppend(const TrajOptProblemTpl<Scalar> &problem,
                                       shared_ptr<StageDataTpl<Scalar>> data) {
  rotate_vec_left(problem_data.stage_data);
  problem_data.stage_data[nsteps - 1] = data;

  const StageModel &stage = *problem.stages_[nsteps - 1];

  if (!problem.checkIntegrity())
    ALIGATOR_RUNTIME_ERROR("Problem failed integrity check.");

  rotate_vec_left(trial_xs, 1);
  rotate_vec_left(trial_us, 1);
  rotate_vec_left(trial_vs, 0, 1);
  trial_vs[nsteps - 1].setZero(stage.nc());
  rotate_vec_left(trial_lams, 1);

  rotate_vec_left(prev_xs);
  rotate_vec_left(prev_us);
  rotate_vec_left(prev_vs, 0, 1);
  prev_vs[nsteps - 1].setZero(stage.nc());

  vs_plus = vs_pdal = trial_vs;
  lams_plus = trial_lams;

  dyn_slacks = trial_lams; // same dimensions
  stage_cstr_violations.setZero();
  stage_infeasibilities = trial_vs;

  shifted_constraints = prev_vs;
  rotate_vec_left(lqr_problem.stages, 0, 1);
  // move assignment, will perform a copy if necessary
  lqr_problem.stages[nsteps - 1] =
      KnotType(uint(stage.ndx1()), uint(stage.nu()), uint(stage.nc()),
               lqr_problem.get_allocator());

  rotate_vec_left(cstr_product_sets, 0, 1);
  cstr_product_sets[nsteps - 1] =
      ConstraintSetProduct(getConstraintProductSet(stage.constraints_));
  rotate_vec_left(cstr_proj_jacs, 0, 1);
  cstr_proj_jacs[nsteps - 1] =
      BlkJacobianType(stage.constraints_.dims(), {stage.ndx1(), stage.nu()});
  rotate_vec_left(active_constraints, 0, 1);
  active_constraints[nsteps - 1].setZero(stage.nc());

  // initial condition
  std::tie(dxs, dus, dvs, dlams) =
      gar::lqrInitializeSolution(lqr_problem); // lqr subproblem variables

  Lxs = dxs;
  Lus = dus;
  Lvs = dvs;
  cstr_lx_corr = Lxs;
  cstr_lu_corr = Lus;

  stage_inner_crits.setZero();
  state_dual_infeas.setZero();
  control_dual_infeas.setZero();
}

} // namespace aligator
