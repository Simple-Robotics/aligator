#pragma once

#include "proxddp/core/explicit-dynamics.hpp"

#include "proxddp/utils/newton-raphson.hpp"
#include "proxddp/utils/exceptions.hpp"

#include <type_traits>

namespace proxddp {

namespace internal {

struct __forward_dyn final {

  template <typename T>
  using ConstVectorRef = typename math_types<T>::ConstVectorRef;
  template <typename T> using VectorRef = typename math_types<T>::VectorRef;
  template <typename T> using Vector = typename math_types<T>::VectorXs;

  template <typename T>
  void operator()(const DynamicsModelTpl<T> &model, const ConstVectorRef<T> &x,
                  const ConstVectorRef<T> &u, DynamicsDataTpl<T> &data,
                  VectorRef<T> xout, const std::size_t max_iters = 1000,
                  Vector<T> *gap = 0, double EPS = 1e-6) const {
    using ExpModel = ExplicitDynamicsModelTpl<T>;
    using ExpData = ExplicitDynamicsDataTpl<T>;
    using MatrixRef = typename math_types<T>::MatrixRef;

    if (model.is_explicit()) {
      const auto &model_cast = static_cast<const ExpModel &>(model);
      auto &data_cast = static_cast<ExpData &>(data);
      (*this)(model_cast, x, u, data_cast, xout, max_iters, gap);
    } else {
      // create NewtonRaph algo's data
      Vector<T> dx0buf(model.ndx2);
      dx0buf.setZero();
      typename NewtonRaphson<T>::DataView nr_data{data.value_, dx0buf,
                                                  data.Jy_};
      NewtonRaphson<T>::run(
          model.space_next(),
          [&](const ConstVectorRef<T> &xnext, VectorRef<T> out) {
            model.evaluate(x, u, xnext, data);
            out = data.value_;
            if (gap != 0)
              out += *gap;
          },
          [&](const ConstVectorRef<T> &xnext, MatrixRef Jout) {
            model.computeJacobians(x, u, xnext, data);
            Jout = data.Jy_;
          },
          x, xout, nr_data, EPS, max_iters);
    }
  }

  template <typename T>
  void operator()(const ExplicitDynamicsModelTpl<T> &model,
                  const ConstVectorRef<T> &x, const ConstVectorRef<T> &u,
                  ExplicitDynamicsDataTpl<T> &data, VectorRef<T> xout,
                  const std::size_t = 0, Vector<T> *gap = 0) const {
    model.forward(x, u, data);
    xout = data.xnext_;
    if (gap != 0) {
      model.space_next().integrate(xout, *gap, xout);
    }
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
constexpr internal::__forward_dyn forwardDynamics{};

/// @brief Perform a rollout of the supplied dynamical models.
template <typename Scalar>
typename math_types<Scalar>::VectorOfVectors
rollout(const std::vector<shared_ptr<DynamicsModelTpl<Scalar>>> &dyn_models,
        const typename math_types<Scalar>::VectorXs &x0,
        const typename math_types<Scalar>::VectorOfVectors &us,
        typename math_types<Scalar>::VectorOfVectors &xout) {
  using Data = DynamicsDataTpl<Scalar>;
  const std::size_t N = us.size();
  if (dyn_models.size() != N) {
    PROXDDP_RUNTIME_ERROR(
        "Number of controls should be the same as number of dynamical models!");
  }
  xout.resize(N + 1);
  xout[0] = x0;

  for (std::size_t i = 0; i < N; i++) {
    shared_ptr<Data> data = dyn_models[i]->createData();
    const ManifoldAbstractTpl<Scalar> &space = dyn_models[i]->space();
    xout.push_back(space.neutral());
    forwardDynamics(*dyn_models[i], xout[i], us[i], *data, xout[i + 1]);
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
    forwardDynamics(dyn_model, xs[i], us[i], *data, xs[i + 1]);
  }
  return xs;
}

/// @copybrief  rollout()
/// @details    This overload applies to explicit forward dynamics.
template <typename Scalar>
void rollout(
    const std::vector<shared_ptr<ExplicitDynamicsModelTpl<Scalar>>> &dyn_models,
    const typename math_types<Scalar>::VectorXs &x0,
    const typename math_types<Scalar>::VectorOfVectors &us,
    typename math_types<Scalar>::VectorOfVectors &xout) {
  using DataType = ExplicitDynamicsDataTpl<Scalar>;
  const std::size_t N = us.size();
  xout.resize(N + 1);
  xout[0] = x0;
  if (dyn_models.size() != N) {
    PROXDDP_RUNTIME_ERROR(
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
rollout(const std::vector<shared_ptr<C<Scalar>>> &dms,
        const typename math_types<Scalar>::VectorXs &x0,
        const typename math_types<Scalar>::VectorOfVectors &us) {
  const std::size_t N = us.size();
  typename math_types<Scalar>::VectorOfVectors xout;
  xout.reserve(N + 1);
  rollout(dms, x0, us, xout);
  return xout;
}

} // namespace proxddp
