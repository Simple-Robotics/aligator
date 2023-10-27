/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/function-abstract.hpp"
#include "proxddp/core/unary-function.hpp"

namespace proxddp {

namespace detail {
template <typename Base> struct slice_impl_tpl;
}

template <typename Scalar> struct FunctionSliceDataTpl;

/// @brief  Represents a function of which the output is a subset of another
/// function, for instance \f$x \mapsto f_\{0, 1, 3\}(x) \f$ where \f$f\f$ is
/// given.
template <typename Scalar, typename Base = StageFunctionTpl<Scalar>>
struct FunctionSliceXprTpl : Base, detail::slice_impl_tpl<Base> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using BaseData = StageFunctionDataTpl<Scalar>;
  using SliceImpl = detail::slice_impl_tpl<StageFunctionTpl<Scalar>>;
  using Data = FunctionSliceDataTpl<Scalar>;

  FunctionSliceXprTpl(shared_ptr<Base> func, std::vector<int> const &indices)
      : Base(func->ndx1, func->nu, func->ndx2, (int)indices.size()),
        SliceImpl(func, indices) {}

  FunctionSliceXprTpl(shared_ptr<Base> func, const int idx)
      : FunctionSliceXprTpl(func, std::vector<int>{idx}) {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &y, BaseData &data) const override {

    this->evaluate_impl(data, x, u, y);
  }

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &y,
                        BaseData &data) const override {
    this->computeJacobians_impl(data, x, u, y);
  }

  void computeVectorHessianProducts(const ConstVectorRef &x,
                                    const ConstVectorRef &u,
                                    const ConstVectorRef &y,
                                    const ConstVectorRef &lbda,
                                    BaseData &data) const override {
    this->computeVectorHessianProducts_impl(data, lbda, x, u, y);
  }

  shared_ptr<BaseData> createData() const override {
    return std::make_shared<Data>(*this);
  }
};

template <typename Scalar>
struct FunctionSliceXprTpl<Scalar, UnaryFunctionTpl<Scalar>>
    : UnaryFunctionTpl<Scalar>,
      detail::slice_impl_tpl<UnaryFunctionTpl<Scalar>> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using BaseData = StageFunctionDataTpl<Scalar>;
  using Data = FunctionSliceDataTpl<Scalar>;
  PROXDDP_UNARY_FUNCTION_INTERFACE(Scalar);
  using SliceImpl = detail::slice_impl_tpl<UnaryFunctionTpl<Scalar>>;

  FunctionSliceXprTpl(shared_ptr<Base> func, std::vector<int> const &indices)
      : Base(func->ndx1, func->nu, func->ndx2, (int)indices.size()),
        SliceImpl(func, indices) {}

  FunctionSliceXprTpl(shared_ptr<Base> func, const int idx)
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
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  shared_ptr<BaseData> sub_data;
  VectorXs lbda_sub;

  template <typename Base>
  FunctionSliceDataTpl(FunctionSliceXprTpl<Scalar, Base> const &obj)
      : BaseData(obj.ndx1, obj.nu, obj.ndx2, obj.nr),
        sub_data(obj.func->createData()), lbda_sub(obj.nr) {}
};

namespace detail {
/// @brief Slicing and indexing of a function's output.
template <typename Base> struct slice_impl_tpl {
  using Scalar = typename Base::Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using BaseData = StageFunctionDataTpl<Scalar>;

  using Data = FunctionSliceDataTpl<Scalar>;

  shared_ptr<Base> func;
  /// @brief
  std::vector<int> indices;

  slice_impl_tpl(shared_ptr<Base> func, std::vector<int> const &indices);
  slice_impl_tpl(shared_ptr<Base> func, int idx);

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

} // namespace proxddp

#include "proxddp/modelling/function-xpr-slice.hxx"

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxddp/modelling/function-xpr-slice.txx"
#endif
