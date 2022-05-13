#pragma once

#include "proxddp/core/problem.hpp"


namespace proxddp
{
  
  /// @brief    Results holder struct.
  template<typename _Scalar>
  struct ResultsTpl
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

    /// States
    std::vector<VectorXs> xs_;
    /// Controls
    std::vector<VectorXs> us_;
    /// Problem Lagrange multipliers
    std::vector<VectorXs> lams_;
    /// Dynamics' co-states
    std::vector<VectorRef> co_state_;

    /// @brief    Create the results struct from a problem (ShootingProblemTpl) instance.
    ResultsTpl(const ShootingProblemTpl<Scalar>& problem)
    {

      const std::size_t nsteps = problem.numSteps();
      xs_.reserve(nsteps + 1);
      us_.reserve(nsteps);
      lams_.reserve(nsteps);
      co_state_.reserve(nsteps);
      std::size_t i = 0;
      int nx;
      int nu;
      int ndual;
      for (i = 0; i < nsteps; i++)
      {
        const StageModelTpl<Scalar>& stage = problem.stages_[i];
        nx = stage.xspace1_.nx();
        nu = stage.nu();
        ndual = stage.numDual();
        xs_.push_back(VectorXs::Zero(nx));
        us_.push_back(VectorXs::Zero(nu));
        lams_.push_back(VectorXs::Zero(ndual));
        co_state_.push_back(lams_[i].head(stage.dyn_model_.nr));
      }
      xs_.push_back(VectorXs::Zero(problem.stages_[nsteps - 1].xspace2_.ndx()));
    }

  };

} // namespace proxddp

