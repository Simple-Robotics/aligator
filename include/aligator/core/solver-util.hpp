/// @file solver-util.hpp
/// @brief Common utilities for all solvers.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/fwd.hpp"
#include "aligator/core/traj-opt-problem.hpp"
#include "aligator/utils/exceptions.hpp"

namespace aligator {

/// @brief Default-intialize a trajectory to the neutral states for each state
/// space at each stage.
template <typename Scalar>
void xs_default_init(const TrajOptProblemTpl<Scalar> &problem,
                     std::vector<typename math_types<Scalar>::VectorXs> &xs) {
  const std::size_t nsteps = problem.numSteps();
  xs.resize(nsteps + 1);
  if (problem.initCondIsStateError()) {
    xs[0] = problem.getInitState();
  } else {
    if (problem.stages_.size() > 0) {
      xs[0] = problem.stages_[0]->xspace().neutral();
    } else {
      ALIGATOR_RUNTIME_ERROR(
          "The problem should have either a StateErrorResidual as an initial "
          "condition or at least one stage.");
    }
  }
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModelTpl<Scalar> &sm = *problem.stages_[i];
    xs[i + 1] = sm.xspace_next().neutral();
  }
}

/// @brief Default-initialize a controls trajectory from the neutral element of
/// each control space.
template <typename Scalar>
void us_default_init(const TrajOptProblemTpl<Scalar> &problem,
                     std::vector<typename math_types<Scalar>::VectorXs> &us) {
  const std::size_t nsteps = problem.numSteps();
  us.resize(nsteps);
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModelTpl<Scalar> &sm = *problem.stages_[i];
    us[i] = sm.uspace().neutral();
  }
}

/// @brief Check the input state-control trajectory is a consistent warm-start
/// for the output.
template <typename Scalar>
void check_trajectory_and_assign(
    const TrajOptProblemTpl<Scalar> &problem,
    const typename math_types<Scalar>::VectorOfVectors &xs_init,
    const typename math_types<Scalar>::VectorOfVectors &us_init,
    typename math_types<Scalar>::VectorOfVectors &xs_out,
    typename math_types<Scalar>::VectorOfVectors &us_out) {
  const std::size_t nsteps = problem.numSteps();
  xs_out.reserve(nsteps + 1);
  us_out.reserve(nsteps);
  if (xs_init.size() == 0) {
    xs_default_init(problem, xs_out);
  } else {
    if (xs_init.size() != (nsteps + 1)) {
      ALIGATOR_RUNTIME_ERROR("warm-start for xs has wrong size!");
    }
    xs_out = xs_init;
  }
  if (us_init.size() == 0) {
    us_default_init(problem, us_out);
  } else {
    if (us_init.size() != nsteps) {
      ALIGATOR_RUNTIME_ERROR("warm-start for us has wrong size!");
    }
    us_out = us_init;
  }
}

/// @brief  Compute the derivatives of the problem Lagrangian.
template <typename Scalar>
void computeLagrangianDerivatives(
    const TrajOptProblemTpl<Scalar> &problem, WorkspaceTpl<Scalar> &workspace,
    const typename math_types<Scalar>::VectorOfVectors &lams) {
  using TrajOptData = TrajOptDataTpl<Scalar>;
  using ConstraintStack = ConstraintStackTpl<Scalar>;
  using StageFunctionData = StageFunctionDataTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;
  using StageModel = StageModelTpl<Scalar>;
  using StageData = StageDataTpl<Scalar>;
  using VectorXs = typename math_types<Scalar>::VectorXs;
  using ConstVectorRef = typename math_types<Scalar>::ConstVectorRef;

  TrajOptData const &pd = workspace.problem_data;
  std::vector<VectorXs> &Lxs = workspace.Lxs_;
  std::vector<VectorXs> &Lus = workspace.Lus_;

  std::size_t nsteps = workspace.nsteps;

  math::setZero(Lxs);
  math::setZero(Lus);
  {
    StageFunctionData const &ind = pd.getInitData();
    Lxs[0] += ind.Jx_.transpose() * lams[0];
  }

  {
    CostData const &cdterm = *pd.term_cost_data;
    Lxs[nsteps] = cdterm.Lx_;
    ConstraintStack const &stack = problem.term_cstrs_;
    VectorXs const &lamN = lams.back();
    for (std::size_t j = 0; j < stack.size(); j++) {
      StageFunctionData const &cstr_data = *pd.term_cstr_data[j];
      auto lam_j = stack.constSegmentByConstraint(lamN, j);
      Lxs[nsteps] += cstr_data.Jx_.transpose() * lam_j;
    }
  }

  for (std::size_t i = 0; i < nsteps; i++) {
    StageModel const &sm = *problem.stages_[i];
    StageData const &sd = pd.getStageData(i);
    ConstraintStack const &stack = sm.constraints_;
    Lxs[i] += sd.cost_data->Lx_;
    Lus[i] += sd.cost_data->Lu_;

    assert(sd.constraint_data.size() == sm.numConstraints());

    for (std::size_t j = 0; j < stack.size(); j++) {
      StageFunctionData const &cstr_data = *sd.constraint_data[j];
      ConstVectorRef lam_j = stack.constSegmentByConstraint(lams[i + 1], j);
      Lxs[i] += cstr_data.Jx_.transpose() * lam_j;
      Lus[i] += cstr_data.Ju_.transpose() * lam_j;

      assert((i + 1) <= nsteps);
      // add contribution to the next node
      Lxs[i + 1] += cstr_data.Jy_.transpose() * lam_j;
    }
  }
}

} // namespace aligator
