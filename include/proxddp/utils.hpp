#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/dynamics.hpp"
#include "proxddp/core/explicit-dynamics.hpp"

#include <stdexcept>

namespace proxddp {
/// @brief   Perform a rollout of the controlled trajectory.
/// @todo    Implement for generic DynamicsModelTpl.
template <typename Scalar>
typename math_types<Scalar>::VectorOfVectors
rollout(const std::vector<const DynamicsModelTpl<Scalar> *> &dyn_models,
        const typename math_types<Scalar>::VectorXs &x0,
        const typename math_types<Scalar>::VectorOfVectors &us);

/// @copybrief  rollout()
/// @details    This overload applies to explicit forward dynamics.
template <typename Scalar>
typename math_types<Scalar>::VectorOfVectors
rollout(const std::vector<const ExplicitDynamicsModelTpl<Scalar> *> &dyn_models,
        const typename math_types<Scalar>::VectorXs &x0,
        const typename math_types<Scalar>::VectorOfVectors &us) {
  using VectorXs = typename math_types<Scalar>::VectorXs;
  using DataType = ExplicitDynamicsDataTpl<Scalar>;
  std::vector<VectorXs> xs{x0};
  const std::size_t N = us.size();
  xs.reserve(N + 1);
  if (dyn_models.size() != N) {
    throw std::domain_error(
        "Number of controls should be the same as number of dynamical models!");
  }

  for (std::size_t i = 0; i < N; i++) {
    shared_ptr<DataType> data =
        std::static_pointer_cast<DataType>(dyn_models[i]->createData());
    xs.push_back(dyn_models[i]->next_state_->neutral());
    dyn_models[i]->forward(xs[i], us[i], *data);
    xs.push_back(data->xnext_);
  }

  return xs;
}

/// @copybrief rollout() Single model version.
/// @details  This version rolls out a single model by copying it.
template <typename Scalar>
typename math_types<Scalar>::VectorOfVectors
rollout(const ExplicitDynamicsModelTpl<Scalar> &dyn_model,
        const typename math_types<Scalar>::VectorXs &x0,
        const typename math_types<Scalar>::VectorOfVectors &us) {
  using VectorXs = typename math_types<Scalar>::VectorXs;
  using DataType = ExplicitDynamicsDataTpl<Scalar>;
  const std::size_t N = us.size();
  std::vector<VectorXs> xs{x0};
  xs.reserve(N + 1);

  shared_ptr<DataType> data =
      std::static_pointer_cast<DataType>(dyn_model.createData());
  for (std::size_t i = 0; i < N; i++) {
    dyn_model.forward(xs[i], us[i], *data);
    xs.push_back(data->xnext_);
  }
  return xs;
}

} // namespace proxddp
