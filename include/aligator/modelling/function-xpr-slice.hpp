/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/function-abstract.hpp"
#include "aligator/core/unary-function.hpp"
#include "aligator/third-party/polymorphic_cxx14.h"

namespace aligator {

namespace detail {
template <typename Base> struct slice_impl_tpl;
}

template <typename Scalar> struct FunctionSliceDataTpl;

/// @brief  Represents a function of which the output is a subset of another
/// function, for instance \f$x \mapsto f_\{0, 1, 3\}(x) \f$ where \f$f\f$ is
/// given.
template <typename Scalar, typename Base = StageFunctionTpl<Scalar>>
struct FunctionSliceXprTpl;

template <typename Scalar>
struct FunctionSliceXprTpl<Scalar, StageFunctionTpl<Scalar>>
    : StageFunctionTpl<Scalar>,
      detail::slice_impl_tpl<StageFunctionTpl<Scalar>> {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using BaseData = StageFunctionDataTpl<Scalar>;
  using SliceImpl = detail::slice_impl_tpl<StageFunctionTpl<Scalar>>;
  using Data = FunctionSliceDataTpl<Scalar>;

  FunctionSliceXprTpl(xyz::polymorphic<Base> func,
                      std::vector<int> const &indices)
      : Base(func->ndx1, func->nu, (int)indices.size())
      , SliceImpl(func, indices) {}

  FunctionSliceXprTpl(xyz::polymorphic<Base> func, const int idx)
      : FunctionSliceXprTpl(func, std::vector<int>{idx}) {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) const override {

    this->evaluate_impl(data, x, u);
  }

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        BaseData &data) const override {
    this->computeJacobians_impl(data, x, u);
  }

  void computeVectorHessianProducts(const ConstVectorRef &x,
                                    const ConstVectorRef &u,
                                    const ConstVectorRef &lbda,
                                    BaseData &data) const override {
    this->computeVectorHessianProducts_impl(data, lbda, x, u);
  }

  shared_ptr<BaseData> createData() const override {
    return std::make_shared<Data>(*this);
  }
};

template <typename Scalar>
struct FunctionSliceXprTpl<Scalar, UnaryFunctionTpl<Scalar>>
    : UnaryFunctionTpl<Scalar>,
      detail::slice_impl_tpl<UnaryFunctionTpl<Scalar>> {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using BaseData = StageFunctionDataTpl<Scalar>;
  using Data = FunctionSliceDataTpl<Scalar>;
  ALIGATOR_UNARY_FUNCTION_INTERFACE(Scalar);
  using SliceImpl = detail::slice_impl_tpl<UnaryFunctionTpl<Scalar>>;

  FunctionSliceXprTpl(xyz::polymorphic<Base> func,
                      std::vector<int> const &indices)
      : Base(func->ndx1, func->nu, (int)indices.size())
      , SliceImpl(func, indices) {}

  FunctionSliceXprTpl(xyz::polymorphic<Base> func, const int idx)
      : FunctionSliceXprTpl(func, std::vector<int>{idx}) {}

  void evaluate(const ConstVectorRef &x, BaseData &data) const override {

    this->evaluate_impl(data, x);
  }

  void computeJacobians(const ConstVectorRef &x,
                        BaseData &data) const override {
    this->computeJacobians_impl(data, x);
  }

  void computeVectorHessianProducts(const ConstVectorRef &x,
                                    const ConstVectorRef &lbda,
                                    BaseData &data) const override {
    this->computeVectorHessianProducts_impl(data, lbda, x);
  }

  shared_ptr<BaseData> createData() const override {
    return std::make_shared<Data>(*this);
  }
};

template <typename Scalar>
struct FunctionSliceDataTpl : StageFunctionDataTpl<Scalar> {
  /// @brief Base residual's data object.
  using BaseData = StageFunctionDataTpl<Scalar>;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  shared_ptr<BaseData> sub_data;
  VectorXs lbda_sub;

  template <typename Base>
  FunctionSliceDataTpl(FunctionSliceXprTpl<Scalar, Base> const &obj)
      : BaseData(obj.ndx1, obj.nu, obj.nr)
      , sub_data(obj.func->createData())
      , lbda_sub(obj.nr) {}
};

namespace detail {
/// @brief Slicing and indexing of a function's output.
template <typename Base> struct slice_impl_tpl {
  using Scalar = typename Base::Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using BaseData = StageFunctionDataTpl<Scalar>;

  using Data = FunctionSliceDataTpl<Scalar>;

  xyz::polymorphic<Base> func;
  /// @brief
  std::vector<int> indices;

  slice_impl_tpl(xyz::polymorphic<Base> func, std::vector<int> const &indices);
  slice_impl_tpl(xyz::polymorphic<Base> func, int idx);

protected:
  template <typename... Args>
  void evaluate_impl(BaseData &data, Args &&...args) const;

  template <typename... Args>
  void computeJacobians_impl(BaseData &data, Args &&...args) const;

  template <typename... Args>
  void computeVectorHessianProducts_impl(BaseData &data,
                                         const ConstVectorRef &lbda,
                                         Args &&...args) const;
};
} // namespace detail

} // namespace aligator

#include "aligator/modelling/function-xpr-slice.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/modelling/function-xpr-slice.txx"
#endif
