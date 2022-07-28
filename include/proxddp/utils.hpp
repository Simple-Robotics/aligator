#pragma once

#include "proxddp/core/dynamics.hpp"
#include "proxddp/core/explicit-dynamics.hpp"

#include "proxddp/utils/newton-raphson.hpp"
#include "proxddp/utils/exceptions.hpp"

namespace proxddp {

namespace internal {

struct __forward_dyn {
  double EPS = 1e-6;
  template <typename T>
  void operator()(const ManifoldAbstractTpl<T> &space,
                  const DynamicsModelTpl<T> &model,
                  const typename math_types<T>::ConstVectorRef &x,
                  const typename math_types<T>::ConstVectorRef &u,
                  DynamicsDataTpl<T> &data,
                  typename math_types<T>::VectorRef xout) const {
    const auto *model_ptr_cast =
        dynamic_cast<const ExplicitDynamicsModelTpl<Scalar> *>(&model);
    const auto *data_ptr_cast =
        dynamic_cast<ExplicitDynamicsDataTpl<Scalar> *>(&data);
    bool check = (model_ptr_cast != nullptr) && (data_ptr_cast != nullptr);
    if (check) {
      // safely deref to an ExplicitDynamicsModelTpl
      this->operator()(*model_ptr_cast, x, u, *data_ptr_cast, xout);
    } else {
      using ConstVectorRef = typename math_types<T>::ConstVectorRef;
      auto fun = [&](const ConstVectorRef &xnext) {
        model.evaluate(x, u, xnext, data);
        return data.value_;
      };
      auto Jfun = [&](const ConstVectorRef &xnext) {
        model.computeJacobians(x, u, xnext, data);
        return data.Jy_;
      };
      NewtonRaphson<T>::run(space, fun, Jfun, x, xout, EPS);
    }
  }

  /// Override; falls back to the standard behaviour.
  template <typename T>
  void operator()(const ManifoldAbstractTpl<T> &,
                  const ExplicitDynamicsModelTpl<T> &model,
                  const typename math_types<T>::ConstVectorRef &x,
                  const typename math_types<T>::ConstVectorRef &u,
                  ExplicitDynamicsDataTpl<T> &data,
                  typename math_types<T>::VectorRef xout) {
    model.forward(x, u, data);
    xout = data.xnext_;
  }
};

} // namespace internal

/**
 * @brief    Evaluates the forward map for a discrete dynamics model, implicit
 * or explicit.
 * @details  If the given DynamicsModelTpl can be safely downcast to an explicit
 * dynamics type then this function will use the
 * ExplicitDynamicsModelTpl::forward() method.
 */
inline constexpr internal::__forward_dyn forwardDynamics{};

/// @brief   Perform a rollout of the controlled trajectory.
/// @todo    Implement for generic DynamicsModelTpl.
template <typename Scalar>
typename math_types<Scalar>::VectorOfVectors
rollout(const ManifoldAbstractTpl<Scalar> &space,
        const std::vector<const DynamicsModelTpl<Scalar> *> &dyn_models,
        const typename math_types<Scalar>::VectorXs &x0,
        const typename math_types<Scalar>::VectorOfVectors &us) {
  using VectorXs = typename math_types<Scalar>::VectorXs;
  using Data = DynamicsDataTpl<Scalar>;
  const std::size_t N = us.size();
  if (dyn_models.size() != N) {
    proxddp_runtime_error(
        "Number of controls should be the same as number of dynamical models!");
  }

  std::vector<VectorXs> xs{x0};
  xs.reserve(N + 1);

  for (std::size_t i = 0; i < N; i++) {
    shared_ptr<Data> data = dyn_models[i]->createData();
    xs.push_back(space.neutral());
    forwardDynamics(space, *dyn_models[i], xs[i], us[i], *data, xs[i + 1]);
  }
  return xs;
}

/// @copybrief rollout()
template <typename Scalar>
typename math_types<Scalar>::VectorOfVectors
rollout(const ManifoldAbstractTpl<Scalar> &space,
        const DynamicsModelTpl<Scalar> &dyn_model,
        const typename math_types<Scalar>::VectorXs &x0,
        const typename math_types<Scalar>::VectorOfVectors &us) {
  using VectorXs = typename math_types<Scalar>::VectorXs;

  const std::size_t N = us.size();
  std::vector<VectorXs> xs{x0};
  xs.reserve(N + 1);
  shared_ptr<DynamicsDataTpl<Scalar>> data = dyn_model.createData();

  for (std::size_t i = 0; i < N; i++) {
    xs.push_back(space.neutral());
    forwardDynamics(space, dyn_model, xs[i], us[i], *data, xs[i + 1]);
  }
  return xs;
}

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
    proxddp_runtime_error(
        "Number of controls should be the same as number of dynamical models!");
  }

  for (std::size_t i = 0; i < N; i++) {
    shared_ptr<DataType> data =
        std::static_pointer_cast<DataType>(dyn_models[i]->createData());
    dyn_models[i]->forward(xs[i], us[i], *data);
    xs.push_back(data->xnext_);
  }

  return xs;
}

/// @copybrief rollout() Rolls out a single ExplicitDynamicsModelTpl.
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
