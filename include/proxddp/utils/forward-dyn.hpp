#pragma once

#include "proxddp/core/explicit-dynamics.hpp"
#include "proxddp/utils/newton-raphson.hpp"

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
          x, xout, data.value_, dx0buf, data.Jy_, EPS, max_iters);
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

} // namespace proxddp
