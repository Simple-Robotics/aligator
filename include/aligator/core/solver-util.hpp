/// @file solver-util.hpp
/// @brief Common utilities for all solvers.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/fwd.hpp"
#include "aligator/core/traj-opt-problem.hpp"
#include "aligator/utils/exceptions.hpp"
#include "aligator/gar/blk-matrix.hpp"

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
template <typename Scalar> struct LagrangianDerivatives {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using TrajOptProblem = TrajOptProblemTpl<Scalar>;
  using TrajOptData = TrajOptDataTpl<Scalar>;
  using BlkView = BlkMatrix<ConstVectorRef, -1, 1>;

  static void compute(const TrajOptProblem &problem, const TrajOptData &pd,
                      const std::vector<VectorXs> &lams,
                      std::vector<VectorXs> &Lxs, std::vector<VectorXs> &Lus) {
    using ConstraintStack = ConstraintStackTpl<Scalar>;
    using StageFunctionData = StageFunctionDataTpl<Scalar>;
    using CostData = CostDataAbstractTpl<Scalar>;
    using StageModel = StageModelTpl<Scalar>;
    using StageData = StageDataTpl<Scalar>;
    std::size_t nsteps = problem.numSteps();

    math::setZero(Lxs);
    math::setZero(Lus);
    {
      StageFunctionData const &ind = *pd.init_data;
      Lxs[0] += ind.Jx_.transpose() * lams[0];
    }

    {
      CostData const &cdterm = *pd.term_cost_data;
      Lxs[nsteps] = cdterm.Lx_;
      ConstraintStack const &stack = problem.term_cstrs_;
      if (!stack.empty()) {
        VectorXs const &lamN = lams.back();
        BlkView lview(lamN, stack.getDims());
        for (std::size_t j = 0; j < stack.size(); j++) {
          StageFunctionData const &cstr_data = *pd.term_cstr_data[j];
          Lxs[nsteps] += cstr_data.Jx_.transpose() * lview[j];
        }
      }
    }

    for (std::size_t i = 0; i < nsteps; i++) {
      StageModel const &sm = *problem.stages_[i];
      StageData const &sd = *pd.stage_data[i];
      ConstraintStack const &stack = sm.constraints_;
      Lxs[i] += sd.cost_data->Lx_;
      Lus[i] += sd.cost_data->Lu_;

      assert(sd.constraint_data.size() == sm.numConstraints());

      BlkView lview(lams[i + 1], stack.getDims());
      for (std::size_t j = 0; j < stack.size(); j++) {
        StageFunctionData const &cstr_data = *sd.constraint_data[j];
        Lxs[i] += cstr_data.Jx_.transpose() * lview[j];
        Lus[i] += cstr_data.Ju_.transpose() * lview[j];

        assert((i + 1) <= nsteps);
        // add contribution to the next node
        Lxs[i + 1] += cstr_data.Jy_.transpose() * lview[j];
      }
    }
  }
};
} // namespace aligator
