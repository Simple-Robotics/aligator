#pragma once

#include "./function-xpr-slice.hpp"

namespace aligator {
namespace detail {

template <typename Base>
slice_impl_tpl<Base>::slice_impl_tpl(xyz::polymorphic<Base> func,
                                     std::vector<int> const &indices)
    : func(func)
    , indices(indices) {}

template <typename Base>
slice_impl_tpl<Base>::slice_impl_tpl(xyz::polymorphic<Base> func, int idx)
    : slice_impl_tpl(func, std::vector<int>({idx})) {}

template <typename Base>
template <typename... Args>
void slice_impl_tpl<Base>::evaluate_impl(BaseData &data, Args &&...args) const {
  Data &d = static_cast<Data &>(data);
  assert(d.sub_data != 0);
  BaseData &sub_data = *d.sub_data;
  // evaluate base
  func->evaluate(std::forward<Args>(args)..., sub_data);

  for (std::size_t j = 0; j < indices.size(); j++) {
    int i = indices[j];
    d.value_((long)j) = sub_data.value_(i);
  }
}

template <typename Base>
template <typename... Args>
void slice_impl_tpl<Base>::computeJacobians_impl(BaseData &data,
                                                 Args &&...args) const {
  Data &d = static_cast<Data &>(data);
  assert(d.sub_data != 0);
  BaseData &sub_data = *d.sub_data;
  // evaluate base
  func->computeJacobians(std::forward<Args>(args)..., sub_data);

  for (std::size_t j = 0; j < indices.size(); j++) {
    int i = indices[j];
    d.jac_buffer_.row((long)j) = sub_data.jac_buffer_.row(i);
  }
}

template <typename Base>
template <typename... Args>
void slice_impl_tpl<Base>::computeVectorHessianProducts_impl(
    BaseData &data, const ConstVectorRef &lbda, Args &&...args) const {
  Data &d = static_cast<Data &>(data);
  assert(d.sub_data != 0);
  BaseData &sub_data = *d.sub_data;

  d.lbda_sub = lbda;
  for (std::size_t j = 0; j < indices.size(); j++) {
    d.lbda_sub(indices[j]) = 0.;
  }

  func->computeVectorHessianProducts(std::forward<Args>(args)..., d.lbda_sub,
                                     sub_data);
}

} // namespace detail

} // namespace aligator
