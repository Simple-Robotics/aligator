#pragma once

#include "./linear-function-composition.hpp"

namespace aligator {

template <typename Scalar>
void LinearFunctionCompositionTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                    const ConstVectorRef &u,
                                                    BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  func->evaluate(x, u, *d.sub_data);
  data.value_ = b;
  data.value_.noalias() += A * d.sub_data->value_;
}

template <typename Scalar>
void LinearFunctionCompositionTpl<Scalar>::computeJacobians(
    const ConstVectorRef &x, const ConstVectorRef &u, BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  func->computeJacobians(x, u, *d.sub_data);
  data.jac_buffer_.noalias() = A * d.sub_data->jac_buffer_;
}

template <typename Scalar>
void LinearUnaryFunctionCompositionTpl<Scalar>::evaluate(
    const ConstVectorRef &x, BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  func->evaluate(x, *d.sub_data);
  data.value_ = b;
  data.value_.noalias() += A * d.sub_data->value_;
}

template <typename Scalar>
void LinearUnaryFunctionCompositionTpl<Scalar>::computeJacobians(
    const ConstVectorRef &x, BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  func->computeJacobians(x, *d.sub_data);
  data.jac_buffer_.noalias() = A * d.sub_data->jac_buffer_;
}

} // namespace aligator
