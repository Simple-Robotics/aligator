#pragma once

#include "aligator/utils/forward-dyn.hpp"

namespace aligator {

/// @brief Perform a rollout of the supplied dynamical models.
template <typename Scalar>
typename math_types<Scalar>::VectorOfVectors rollout(
    const std::vector<xyz::polymorphic<DynamicsModelTpl<Scalar>>> &dyn_models,
    const typename math_types<Scalar>::VectorXs &x0,
    const typename math_types<Scalar>::VectorOfVectors &us,
    typename math_types<Scalar>::VectorOfVectors &xout) {
  using Data = DynamicsDataTpl<Scalar>;
  const std::size_t N = us.size();
  if (dyn_models.size() != N) {
    ALIGATOR_RUNTIME_ERROR(
        "Number of controls should be the same as number of dynamical models!");
  }
  xout.resize(N + 1);
  xout[0] = x0;

  for (std::size_t i = 0; i < N; i++) {
    shared_ptr<Data> data = dyn_models[i]->createData();
    const ManifoldAbstractTpl<Scalar> &space = dyn_models[i]->space();
    xout.push_back(space.neutral());
    forwardDynamics<Scalar>::run(*dyn_models[i], xout[i], us[i], *data,
                                 xout[i + 1]);
  }
  return xout;
}

/// @copybrief rollout()
template <typename Scalar>
typename math_types<Scalar>::VectorOfVectors
rollout(const DynamicsModelTpl<Scalar> &dyn_model,
        const typename math_types<Scalar>::VectorXs &x0,
        const typename math_types<Scalar>::VectorOfVectors &us) {
  using VectorXs = typename math_types<Scalar>::VectorXs;

  const std::size_t N = us.size();
  std::vector<VectorXs> xs{x0};
  xs.reserve(N + 1);
  shared_ptr<DynamicsDataTpl<Scalar>> data = dyn_model.createData();

  for (std::size_t i = 0; i < N; i++) {
    const ManifoldAbstractTpl<Scalar> &space = dyn_model.space();
    xs.push_back(space.neutral());
    forwardDynamics<Scalar>::run(dyn_model, xs[i], us[i], *data, xs[i + 1]);
  }
  return xs;
}

/// @copybrief  rollout()
/// @details    This overload applies to explicit forward dynamics.
template <typename Scalar>
void rollout(
    const std::vector<xyz::polymorphic<ExplicitDynamicsModelTpl<Scalar>>>
        &dyn_models,
    const typename math_types<Scalar>::VectorXs &x0,
    const typename math_types<Scalar>::VectorOfVectors &us,
    typename math_types<Scalar>::VectorOfVectors &xout) {
  using DataType = ExplicitDynamicsDataTpl<Scalar>;
  const std::size_t N = us.size();
  xout.resize(N + 1);
  xout[0] = x0;
  if (dyn_models.size() != N) {
    ALIGATOR_RUNTIME_ERROR(
        fmt::format("Number of controls ({}) should be the same as number of "
                    "dynamical models ({})!",
                    N, dyn_models.size()));
  }

  for (std::size_t i = 0; i < N; i++) {
    shared_ptr<DataType> data =
        std::static_pointer_cast<DataType>(dyn_models[i]->createData());
    dyn_models[i]->forward(xout[i], us[i], *data);
    xout[i + 1] = data->xnext_;
  }
}

/// @copybrief rollout() Rolls out a single ExplicitDynamicsModelTpl.
template <typename Scalar>
void rollout(const ExplicitDynamicsModelTpl<Scalar> &dyn_model,
             const typename math_types<Scalar>::VectorXs &x0,
             const typename math_types<Scalar>::VectorOfVectors &us,
             typename math_types<Scalar>::VectorOfVectors &xout) {
  using DataType = ExplicitDynamicsDataTpl<Scalar>;
  const std::size_t N = us.size();
  xout.resize(N + 1);
  xout[0] = x0;

  shared_ptr<DataType> data =
      std::static_pointer_cast<DataType>(dyn_model.createData());
  for (std::size_t i = 0; i < N; i++) {
    dyn_model.forward(xout[i], us[i], *data);
    xout[i + 1] = data->xnext_;
  }
}

/// @copybrief rollout(). This variant allocates the output and returns it.
template <template <typename> class C, typename Scalar>
typename math_types<Scalar>::VectorOfVectors
rollout(const C<Scalar> &dms, const typename math_types<Scalar>::VectorXs &x0,
        const typename math_types<Scalar>::VectorOfVectors &us) {
  const std::size_t N = us.size();
  typename math_types<Scalar>::VectorOfVectors xout;
  xout.reserve(N + 1);
  rollout(dms, x0, us, xout);
  return xout;
}

/// @copybrief rollout(). This variant allocates the output and returns it.
template <template <typename> class C, typename Scalar>
typename math_types<Scalar>::VectorOfVectors
rollout(const std::vector<xyz::polymorphic<C<Scalar>>> &dms,
        const typename math_types<Scalar>::VectorXs &x0,
        const typename math_types<Scalar>::VectorOfVectors &us) {
  const std::size_t N = us.size();
  typename math_types<Scalar>::VectorOfVectors xout;
  xout.reserve(N + 1);
  rollout(dms, x0, us, xout);
  return xout;
}

} // namespace aligator
