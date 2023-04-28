#pragma once

#include "./function-xpr-slice.hpp"

namespace proxddp {

template <typename Scalar>
FunctionSliceXprTpl<Scalar>::FunctionSliceXprTpl(
    shared_ptr<Base> func, std::vector<int> const &indices)
    : Base(func->ndx1, func->nu, func->ndx2, (int)indices.size()), func(func),
      indices(indices) {}

template <typename Scalar>
FunctionSliceXprTpl<Scalar>::FunctionSliceXprTpl(shared_ptr<Base> func, int idx)
    : FunctionSliceXprTpl(func, std::vector<int>({idx})) {}

template <typename Scalar>
void FunctionSliceXprTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                           const ConstVectorRef &u,
                                           const ConstVectorRef &y,
                                           BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  assert(d.sub_data != 0);
  BaseData &sub_data = *d.sub_data;
  // evaluate base
  func->evaluate(x, u, y, sub_data);

  for (std::size_t j = 0; j < indices.size(); j++) {
    int i = indices[j];
    d.value_((long)j) = sub_data.value_(i);
  }
}

template <typename Scalar>
void FunctionSliceXprTpl<Scalar>::computeJacobians(const ConstVectorRef &x,
                                                   const ConstVectorRef &u,
                                                   const ConstVectorRef &y,
                                                   BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  assert(d.sub_data != 0);
  BaseData &sub_data = *d.sub_data;
  // evaluate base
  func->computeJacobians(x, u, y, sub_data);

  for (std::size_t j = 0; j < indices.size(); j++) {
    int i = indices[j];
    d.jac_buffer_.row((long)j) = sub_data.jac_buffer_.row(i);
  }
}

template <typename Scalar>
void FunctionSliceXprTpl<Scalar>::computeVectorHessianProducts(
    const ConstVectorRef &x, const ConstVectorRef &u, const ConstVectorRef &y,
    const ConstVectorRef &lbda, BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  assert(d.sub_data != 0);
  BaseData &sub_data = *d.sub_data;

  d.lbda_sub = lbda;
  for (std::size_t j = 0; j < indices.size(); j++) {
    d.lbda_sub(indices[j]) = 0.;
  }

  func->computeVectorHessianProducts(x, u, y, d.lbda_sub, sub_data);
}

template <typename Scalar>
auto FunctionSliceXprTpl<Scalar>::createData() const -> shared_ptr<BaseData> {
  return std::make_shared<Data>(*this);
}
} // namespace proxddp
