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
