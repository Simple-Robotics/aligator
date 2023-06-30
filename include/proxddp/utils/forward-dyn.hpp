#pragma once

#include "proxddp/core/explicit-dynamics.hpp"
#include "proxddp/utils/newton-raphson.hpp"
#include <boost/optional.hpp>

namespace proxddp {

/**
 * @brief    Evaluates the forward map for a discrete dynamics model, implicit
 * or explicit.
 * @details  If the given DynamicsModelTpl can be safely downcast to an explicit
 * dynamics type then this function will use the
 * ExplicitDynamicsModelTpl::forward() method.
 */
template <typename T> struct forwardDynamics {

  using VectorXs = typename math_types<T>::VectorXs;
  using VectorRef = typename math_types<T>::VectorRef;
  using ConstVectorRef = typename math_types<T>::ConstVectorRef;
  using MatrixRef = typename math_types<T>::MatrixRef;

  static void run(const DynamicsModelTpl<T> &model, const ConstVectorRef &x,
                  const ConstVectorRef &u, DynamicsDataTpl<T> &data,
                  VectorRef xout,
                  const boost::optional<ConstVectorRef> &gap = boost::none,
                  const uint max_iters = 1000, const T EPS = 1e-6) {
    using ExpModel = ExplicitDynamicsModelTpl<T>;
    using ExpData = ExplicitDynamicsDataTpl<T>;

    if (model.is_explicit()) {
      const ExpModel &model_cast = static_cast<const ExpModel &>(model);
      ExpData &data_cast = static_cast<ExpData &>(data);
      run(model_cast, x, u, data_cast, xout, gap);
    } else {
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
          x, xout, data.value_, dx0buf, data.Jy_, EPS, max_iters);
    }
  }

  static void run(const ExplicitDynamicsModelTpl<T> &model,
                  const ConstVectorRef &x, const ConstVectorRef &u,
                  ExplicitDynamicsDataTpl<T> &data, VectorRef xout,
                  const boost::optional<ConstVectorRef> &gap = boost::none) {
    model.forward(x, u, data);
    xout = data.xnext_;
    if (gap.has_value()) {
      model.space_next().integrate(xout, *gap, xout);
    }
  }
};

} // namespace proxddp
