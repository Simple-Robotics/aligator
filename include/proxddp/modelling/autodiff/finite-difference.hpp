/// @file   Helper structs to define derivatives on a function
///         through finite differences.
#pragma once

#include "proxddp/core/function-abstract.hpp"

namespace proxddp {
namespace autodiff {

enum FDLevel {
  TOC1 = 0, ///< Cast to a \f$C^1\f$ function.
  TOC2 = 1  ///< Cast to a \f$C^2\f$ function.
};

/// Type of finite differences: forward, central, or forward.
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
struct finite_difference_impl<_Scalar, TOC1> : virtual StageFunctionTpl<_Scalar> {
  PROXNLP_FUNCTION_TYPEDEFS(_Scalar);
  using Base = StageFunctionTpl<_Scalar>;
  using Base::computeJacobians;

  const ManifoldAbstractTpl<_Scalar> &space;
  const Base &func;
  _Scalar fd_eps;

  finite_difference_impl(const ManifoldAbstractTpl<_Scalar> &space,
                         const Base &func, const _Scalar fd_eps)
      : Base(func.ndx1(), func.nu(), func.ndx2(), func.nr()), space(space),
        func(func), fd_eps(fd_eps) {}

  void computeJacobians(const ConstVectorRef &x, MatrixRef Jout) const override {
    VectorXs ei(func.ndx());
    VectorXs xplus = space.neutral();
    VectorXs xminus = space.neutral();
    ei.setZero();
    for (int i = 0; i < func.ndx(); i++) {
      ei(i) = fd_eps;
      space.integrate(x, ei, xplus);
      space.integrate(x, -ei, xminus);
      Jout.col(i) = (func(xplus) - func(xminus)) / (2 * fd_eps);
      ei(i) = 0.;
    }
  }
};

} // namespace internal

template <typename _Scalar, FDLevel n = TOC1> struct finite_difference_wrapper;

/** @brief    Approximate the derivatives of a given function
 *            using finite differences, to downcast the function to a
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

  PROXNLP_FUNCTION_TYPEDEFS(_Scalar);

  finite_difference_wrapper(const ManifoldAbstractTpl<Scalar> &space,
                            const InputType &func, const Scalar fd_eps)
      : OutType(func.ndx1(), func.nu(), func.ndx2(), func.nr()), Base(space, func, fd_eps) {}

  ReturnType operator()(const ConstVectorRef &x) const { return this->func(x); }
};

} // namespace autodiff
} // namespace proxddp
