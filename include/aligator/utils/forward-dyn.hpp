/// @file forward-dyn.hpp
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/core/explicit-dynamics.hpp"
#include "aligator/utils/newton-raphson.hpp"
#include <optional>

namespace aligator {

template <typename T> struct ForwardDynamicsOptions {
  uint max_iters = 1000u;
  T eps = 1e-6;
};

namespace detail {
template <typename T> struct ForwardDynamics {

  using VectorXs = typename math_types<T>::VectorXs;
  using VectorRef = typename math_types<T>::VectorRef;
  using ConstVectorRef = typename math_types<T>::ConstVectorRef;
  using MatrixRef = typename math_types<T>::MatrixRef;

  static void run(const DynamicsModelTpl<T> &model, const ConstVectorRef &x,
                  const ConstVectorRef &u, DynamicsDataTpl<T> &data,
                  VectorRef xout, const std::optional<ConstVectorRef> &gap,
                  ForwardDynamicsOptions<T> opts) {
    // create NewtonRaph algo's data
    VectorXs dx0buf(model.ndx2);
    dx0buf.setZero();
    NewtonRaphson<T>::run(
        model.space_next(),
        [&](const ConstVectorRef &xnext, VectorRef out) {
          model.evaluate(x, u, xnext, data);
          out = data.value_;
          if (gap.has_value())
            out += *gap;
        },
        [&](const ConstVectorRef &xnext, MatrixRef Jout) {
          model.computeJacobians(x, u, xnext, data);
          Jout = data.Jy_;
        },
        x, xout, data.value_, dx0buf, data.Jy_, opts.eps, opts.max_iters);
  }

  static void run(const ExplicitDynamicsModelTpl<T> &model,
                  const ConstVectorRef &x, const ConstVectorRef &u,
                  ExplicitDynamicsDataTpl<T> &data, VectorRef xout,
                  const std::optional<ConstVectorRef> &gap,
                  ForwardDynamicsOptions<T>) {
    model.forward(x, u, data);
    xout = data.xnext_;
    if (gap.has_value())
      model.space_next().integrate(xout, *gap, xout);
  }
};
} // namespace detail

/// @brief    Evaluates the forward map for a discrete dynamics model, implicit
/// or explicit.
/// @details  If the given model is explicit then this function will just use
/// the ExplicitDynamicsModelTpl::forward() method.
template <typename T, template <typename> class M,
          typename D = typename M<T>::Data>
void forwardDynamics(const M<T> &model,
                     const typename math_types<T>::ConstVectorRef &x,
                     const typename math_types<T>::ConstVectorRef &u, D &data,
                     const typename math_types<T>::VectorRef xout,
                     const std::optional<typename math_types<T>::ConstVectorRef>
                         &gap = std::nullopt,
                     ForwardDynamicsOptions<T> opts = {}) {
  detail::ForwardDynamics<T>::run(model, x, u, data, xout, gap, opts);
}

} // namespace aligator
