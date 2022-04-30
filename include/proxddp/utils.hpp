#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/dynamics.hpp"
#include "proxddp/core/explicit-dynamics.hpp"

namespace proxddp
{
  /// @brief   Perform a rollout of the controlled trajectory.
  /// @todo    Implement for generic DynamicsModelTpl.
  template<typename Scalar>
  typename math_types<Scalar>::VectorOfVectors
  rollout(const std::vector<const DynamicsModelTpl<Scalar>*>& dyn_models,
          const typename math_types<Scalar>::VectorXs& x0,
          const typename math_types<Scalar>::VectorOfVectors& us);

  /// @copybrief  rollout()
  /// @details    This overload applies to explicit forward dynamics.
  template<typename Scalar>
  typename math_types<Scalar>::VectorOfVectors
  rollout(const std::vector<const ExplicitDynamicsModelTpl<Scalar>*>& dyn_models,
          const typename math_types<Scalar>::VectorXs& x0,
          const typename math_types<Scalar>::VectorOfVectors& us)
  {
    typename math_types<Scalar>::VectorOfVectors xs { x0 };
    using VectorXs = typename math_types<Scalar>::VectorXs;
    std::size_t N = us.size();
    xs.reserve(N + 1);
    assert((dyn_models.size() == N) && "Number of controls should be the same as number of dyn models!");

    for (std::size_t i = 0; i < N; i++)
    {
      xs.push_back(VectorXs::Zero(dyn_models[i]->out_space_.nx()));
      dyn_models[i]->forward(xs[i], us[i], xs[i + 1]);
    }

    return xs;
  }

  /// @copybrief rollout() Single model version.
  /// @details  This version rolls out a single model by copying it.
  template<typename Scalar>
  typename math_types<Scalar>::VectorOfVectors
  rollout(const ExplicitDynamicsModelTpl<Scalar>& dyn_model,
          const typename math_types<Scalar>::VectorXs& x0,
          const typename math_types<Scalar>::VectorOfVectors& us)
  {
    const std::size_t N = us.size();
    using C = ExplicitDynamicsModelTpl<Scalar>;
    std::vector<const C*> dyn_models_copies;
    dyn_models_copies.reserve(N);
    for (std::size_t i = 0; i < N; i++)
    {
      dyn_models_copies.push_back(&dyn_model);
    }
    return rollout(dyn_models_copies, x0, us);
  }
  
} // namespace proxddp

