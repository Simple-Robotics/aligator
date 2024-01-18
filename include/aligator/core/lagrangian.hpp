#pragma once

#include "aligator/core/traj-opt-problem.hpp"
#include "aligator/gar/blk-matrix.hpp"

namespace aligator {

/// @brief  Compute the derivatives of the problem Lagrangian.
template <typename Scalar> struct LagrangianDerivatives {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using TrajOptProblem = TrajOptProblemTpl<Scalar>;
  using TrajOptData = TrajOptDataTpl<Scalar>;
  using BlkView = BlkMatrix<ConstVectorRef, -1, 1>;

  static void compute(const TrajOptProblem &problem, const TrajOptData &pd,
                      const std::vector<VectorXs> &lams,
                      const std::vector<VectorXs> &vs,
                      std::vector<VectorXs> &Lxs, std::vector<VectorXs> &Lus) {
    using ConstraintStack = ConstraintStackTpl<Scalar>;
    using StageFunctionData = StageFunctionDataTpl<Scalar>;
    using CostData = CostDataAbstractTpl<Scalar>;
    using StageModel = StageModelTpl<Scalar>;
    using StageData = StageDataTpl<Scalar>;
    const std::size_t nsteps = problem.numSteps();

    ALIGATOR_NOMALLOC_BEGIN;

    math::setZero(Lxs);
    math::setZero(Lus);

    // initial condition
    const StageFunctionData &init_cond = *pd.init_data;
    Lxs[0].noalias() = init_cond.Jx_.transpose() * lams[0];

    for (std::size_t i = 0; i < nsteps; i++) {
      const StageModel &sm = *problem.stages_[i];
      const StageData &sd = *pd.stage_data[i];
      const ConstraintStack &stack = sm.constraints_;
      const StageFunctionData &dd = *sd.dynamics_data;
      Lxs[i].noalias() += sd.cost_data->Lx_ + dd.Jx_.transpose() * lams[i + 1];
      Lus[i].noalias() = sd.cost_data->Lu_ + dd.Ju_.transpose() * lams[i + 1];

      BlkView v_(vs[i], stack.getDims());
      for (std::size_t j = 0; j < stack.size(); j++) {
        const StageFunctionData &cd = *sd.constraint_data[j];
        Lxs[i] += cd.Jx_.transpose() * v_[j];
        Lus[i] += cd.Ju_.transpose() * v_[j];
      }

      Lxs[i + 1].noalias() = dd.Jy_.transpose() * lams[i + 1];
    }

    // terminal node
    {
      const CostData &cdterm = *pd.term_cost_data;
      Lxs[nsteps] += cdterm.Lx_;
      const ConstraintStack &stack = problem.term_cstrs_;
      BlkView vN(vs[nsteps], stack.getDims());
      for (std::size_t j = 0; j < stack.size(); j++) {
        const StageFunctionData &cd = *pd.term_cstr_data[j];
        Lxs[nsteps] += cd.Jx_.transpose() * vN[j];
      }
    }
    ALIGATOR_NOMALLOC_END;
  }
};

} // namespace aligator
