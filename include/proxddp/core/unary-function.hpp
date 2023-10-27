/// @file unary-function.hpp
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/function-abstract.hpp"

namespace proxddp {

/// @brief  Represents unary functions of the form \f$f(x)\f$, with no control
/// (or next-state) arguments.
template <typename _Scalar>
struct UnaryFunctionTpl : StageFunctionTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using Data = StageFunctionDataTpl<Scalar>;

  using Base::Base;

  virtual void evaluate(const ConstVectorRef &x, Data &data) const = 0;
  virtual void computeJacobians(const ConstVectorRef &x, Data &data) const = 0;
  virtual void computeVectorHessianProducts(const ConstVectorRef & /*x*/,
                                            const ConstVectorRef & /*lbda*/,
                                            Data & /*data*/) const {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &,
                const ConstVectorRef &, Data &data) const override {
    this->evaluate(x, data);
  }

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &,
                        const ConstVectorRef &, Data &data) const override {
    this->computeJacobians(x, data);
  }

  void computeVectorHessianProducts(const ConstVectorRef &x,
                                    const ConstVectorRef &,
                                    const ConstVectorRef &,
                                    const ConstVectorRef &lbda,
                                    Data &data) const override {
    this->computeVectorHessianProducts(x, lbda, data);
  }
};

#define PROXDDP_UNARY_FUNCTION_INTERFACE(Scalar)                               \
  using Base = UnaryFunctionTpl<Scalar>;                                       \
  using Base::evaluate;                                                        \
  using Base::computeJacobians;                                                \
  using Base::computeVectorHessianProducts

} // namespace proxddp

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "./unary-function.txx"
#endif
