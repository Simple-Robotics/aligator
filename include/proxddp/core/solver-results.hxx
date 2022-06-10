#pragma once

namespace proxddp
{
  
  template<typename Scalar>
  ResultsTpl<Scalar>::ResultsTpl(const ShootingProblemTpl<Scalar>& problem)
  {

    const std::size_t nsteps = problem.numSteps();
    xs_.reserve(nsteps + 1);
    us_.reserve(nsteps);
    lams_.reserve(nsteps);
    co_state_.reserve(nsteps);
    std::size_t i = 0;
    int nu;
    int ndual;
    for (i = 0; i < nsteps; i++)
    {
      const StageModelTpl<Scalar>& stage = problem.stages_[i];
      nu = stage.nu();
      ndual = stage.numDual();
      xs_.push_back(stage.xspace1_.neutral());
      us_.push_back(VectorXs::Zero(nu));
      lams_.push_back(VectorXs::Ones(ndual));
      co_state_.push_back(lams_[i].head(stage.nx2()));
      if (i == nsteps - 1)
        xs_.push_back(stage.xspace2_.neutral());
    }
    assert(xs_.size() == nsteps + 1);
    assert(us_.size() == nsteps);
  }
} // namespace proxddp

