#pragma once

#include "proxddp/core/function.hpp"

namespace proxddp {

template <typename Scalar>
void StageFunctionTpl<Scalar>::computeVectorHessianProducts(
    const ConstVectorRef &, const ConstVectorRef &, const ConstVectorRef &,
    const ConstVectorRef &, Data &data) const {
  data.vhp_buffer_.setZero();
}

template <typename Scalar>
shared_ptr<FunctionDataTpl<Scalar>>
StageFunctionTpl<Scalar>::createData() const {
  return std::make_shared<Data>(ndx1, nu, ndx2, nr);
}
} // namespace proxddp
