/// @file unary-function.hpp
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/function-abstract.hpp"

namespace aligator {

/// @brief  Represents unary functions of the form \f$f(x)\f$, with no control
/// (or next-state) arguments.
template <typename _Scalar>
struct UnaryFunctionTpl : StageFunctionTpl<_Scalar> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
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

#define ALIGATOR_UNARY_FUNCTION_INTERFACE(Scalar)                              \
  using Base = UnaryFunctionTpl<Scalar>;                                       \
  using Base::evaluate;                                                        \
  using Base::computeJacobians;                                                \
  using Base::computeVectorHessianProducts

#define ALIGATOR_UNARY_FUNCTION_TYPEDEFS(Scalar, _Data)                        \
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);                                           \
  ALIGATOR_UNARY_FUNCTION_INTERFACE(Scalar);                                   \
  using BaseData = StageFunctionDataTpl<Scalar>;                               \
  using Data = _Data<Scalar>;                                                  \
  using CommonModelBuilderContainer = CommonModelBuilderContainerTpl<Scalar>;  \
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>

#define ALIGATOR_UNARY_FUNCTION_DATA_TYPEDEFS(Scalar, _Model)                  \
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);                                           \
  using Model = _Model<Scalar>;                                                \
  using Base = StageFunctionDataTpl<Scalar>;                                   \
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./unary-function.txx"
#endif
