#pragma once

#include "aligator/modelling/centroidal/linear-momentum.hpp"
#include <iostream>
namespace aligator {

template <typename Scalar>
void LinearMomentumResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                 BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.value_ = x.segment(3, 3) - h_ref_;
}

template <typename Scalar>
void LinearMomentumResidualTpl<Scalar>::computeJacobians(
    const ConstVectorRef &x, BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.Jx_.block(0, 3, 3, 3).setIdentity();
}

template <typename Scalar>
LinearMomentumDataTpl<Scalar>::LinearMomentumDataTpl(
    const LinearMomentumResidualTpl<Scalar> *model)
    : Base(model->ndx1, model->nu, model->ndx2, 3) {}

} // namespace aligator
