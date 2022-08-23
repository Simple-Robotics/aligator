/// @file   Helper structs to define derivatives on a function
///         through finite differences.
#pragma once

#include "proxddp/core/function-abstract.hpp"
#include <proxnlp/manifold-base.hpp>

namespace proxddp {
namespace autodiff {

enum FDLevel {
  TOC1 = 0, ///< Cast to a \f$C^1\f$ function.
  TOC2 = 1  ///< Cast to a \f$C^2\f$ function.
};

/// Type of finite differences: forward, central, or backward.
///
enum FDType {
  BACKWARD, ///< Backward finite differences\f$\frac{f_{i} - f_{i-1}}h\f$
  CENTRAL,  ///< Central finite differences\f$\frac{f_{i+1} - f_{i-1}}h\f$
  FORWARD   ///< Forward finite differences\f$\frac{f_{i+1} - f_i}h\f$
};

namespace internal {

// fwd declare the implementation of finite difference algorithms.
template <typename _Scalar, FDLevel n, FDType = CENTRAL>
struct finite_difference_impl;

template <typename _Scalar>
struct finite_difference_impl<_Scalar, TOC1>
    : virtual StageFunctionTpl<_Scalar> {
  PROXNLP_FUNCTION_TYPEDEFS(_Scalar);
  using Base = StageFunctionTpl<_Scalar>;
  using BaseData = FunctionDataTpl<_Scalar>;

  const ManifoldAbstractTpl<_Scalar> &space;
  const Base &func;
  _Scalar fd_eps;

  finite_difference_impl(const ManifoldAbstractTpl<_Scalar> &space,
                         const Base &func, const _Scalar fd_eps)
      : Base(func.ndx1, func.nu, func.ndx2, func.nr), space(space), func(func),
        fd_eps(fd_eps) {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &y, BaseData &data) const override {
    func.evaluate(x, u, y, data);
  }

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &y,
                        BaseData &data) const override {
    VectorXs exi(func.ndx1);
    VectorXs xplus = x;
    VectorXs xminus = x;
    VectorXs vplus(func.nr);
    VectorXs vminus(func.nr);
    exi.setZero();
    for (int i = 0; i < func.ndx1; i++) {
      exi(i) = fd_eps;
      space.integrate(x, exi, xplus);
      space.integrate(x, -exi, xminus);
      func.evaluate(xplus, u, y, data);
      vplus = data.value_;
      func.evaluate(xminus, u, y, data);
      vminus = data.value_;
      data.Jx_.col(i) = (vplus - vminus) / (2 * fd_eps);
      exi(i) = 0.;
    }
    VectorXs eyi(func.ndx2);
    VectorXs yplus = y;
    VectorXs yminus = y;
    eyi.setZero();
    for (int i = 0; i < func.ndx2; i++) {
      eyi(i) = fd_eps;
      space.integrate(y, eyi, yplus);
      space.integrate(y, -eyi, yminus);
      func.evaluate(x, u, yplus, data);
      vplus = data.value_;
      func.evaluate(x, u, yminus, data);
      vminus = data.value_;
      data.Jy_.col(i) = (vplus - vminus) / (2 * fd_eps);
      eyi(i) = 0.;
    }
    VectorXs eui(func.nu);
    VectorXs uplus = u;
    VectorXs uminus = u;
    eui.setZero();
    for (int i = 0; i < func.nu; i++) {
      eui(i) = fd_eps;
      uplus = u + eui;
      uminus = u - eui;
      func.evaluate(x, uplus, y, data);
      vplus = data.value_;
      func.evaluate(x, uminus, y, data);
      vminus = data.value_;
      data.Ju_.col(i) = (vplus - vminus) / (2 * fd_eps);
      eui(i) = 0.;
    }
  }

  void computeVectorHessianProducts(const ConstVectorRef &x,
                                    const ConstVectorRef &u,
                                    const ConstVectorRef &y,
                                    const ConstVectorRef &lbda,
                                    BaseData &data) const {
    func.computeVectorHessianProducts(x, u, y, lbda, data);
  }

  shared_ptr<BaseData> createData() const { return func.createData(); }
};

} // namespace internal

template <typename _Scalar, FDLevel n = TOC1> struct finite_difference_wrapper;

/** @brief    Approximate the derivatives of a given function
 * using finite differences, to downcast the function to a
 * StageFunctionTpl.
 */
template <typename _Scalar>
struct finite_difference_wrapper<_Scalar, TOC1>
    : internal::finite_difference_impl<_Scalar, TOC1> {
  using Scalar = _Scalar;

  using InputType = StageFunctionTpl<Scalar>;
  using OutType = StageFunctionTpl<Scalar>;
  using Base = internal::finite_difference_impl<Scalar, TOC1>;
  using Base::computeJacobians;
  using Base::computeVectorHessianProducts;
  using Base::evaluate;

  PROXNLP_FUNCTION_TYPEDEFS(_Scalar);

  finite_difference_wrapper(const ManifoldAbstractTpl<Scalar> &space,
                            const InputType &func, const Scalar fd_eps)
      : OutType(func.ndx1, func.nu, func.ndx2, func.nr),
        Base(space, func, fd_eps) {}
};

} // namespace autodiff
} // namespace proxddp
