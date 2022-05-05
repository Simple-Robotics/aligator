#pragma once

#include "proxddp/core/problem.hpp"
#include "proxddp/core/solver-workspace.hpp"

#include <fmt/color.h>
#include <fmt/ostream.h>


namespace proxddp
{

  template<typename _Scalar>
  struct ResultsTpl
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar)

    std::vector<VectorXs> xs_;
    std::vector<VectorXs> us_;
    std::vector<VectorXs> lams_;

    ResultsTpl(const ShootingProblemTpl<Scalar>& problem)
    {

      const std::size_t nsteps = problem.numSteps();
      std::size_t i = 0;
      int nx;
      int ndual;
      for (i = 0; i < nsteps; i++)
      {
        const StageModelTpl<Scalar>& stage = problem.stages_[i];
        nx = stage.xspace1_.nx();
        ndual = stage.numDual();
        xs_.push_back(VectorXs::Zero(nx));
        us_.push_back(VectorXs::Zero(stage.nu()));
        lams_.push_back(VectorXs::Zero(ndual));
      }
      xs_.push_back(problem.stages_[nsteps - 1].xspace2_.ndx());
    }

  };

  /// @brief Perform the Riccati forward pass.
  /// This computes the primal-dual step \f$(\delta x,\delta u,\delta\lambda)\f$
  /// @warning This function assumes \f$\delta x_0\f$ has already been computed!
  template<typename Scalar>
  void forward_pass(
    const ShootingProblemTpl<Scalar>& problem,
    WorkspaceTpl<Scalar>& workspace)
  {
    using StageModel = StageModelTpl<Scalar>;
    using VectorXs = typename math_types<Scalar>::VectorXs;
    using MatrixXs = typename math_types<Scalar>::MatrixXs;
    using VectorRef = Eigen::Ref<VectorXs>;
    using MatrixRef = Eigen::Ref<MatrixXs>;

    const std::vector<MatrixXs>& gains = workspace.gains_;
    const std::size_t nsteps = problem.numSteps();
    assert(gains.size() == nsteps);

    int nu;
    int ndual;
    std::size_t i = 0;
    for (i = 0; i < nsteps; i++)
    {
      const StageModel& stage = problem.stages_[i];
      nu = stage.nu();
      ndual = stage.numDual();
      VectorRef feedforward = gains[i].leftCols(1);
      MatrixRef feedback = gains[i].rightCols(stage.ndx1());

      VectorXs pd_step = feedforward + feedback * workspace.dxs_[i];
      workspace.dus_[i] = pd_step.head(nu);
      workspace.dxs_[i + 1] = pd_step.segment(nu, stage.ndx2());
      workspace.dlams_[i] = pd_step.tail(ndual);
    }

  }

  
} // namespace proxddp


