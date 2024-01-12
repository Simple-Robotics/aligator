#pragma once

#include "aligator/modelling/centroidal/angular-momentum.hpp"

namespace aligator {

template <typename Scalar>
void AngularMomentumResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                  BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.value_ = x.segment(6, 3) - L_ref_;
}

template <typename Scalar>
void AngularMomentumResidualTpl<Scalar>::computeJacobians(
    const ConstVectorRef &x, BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.Jx_.bottomRightCorner(3, 3).setIdentity();
}

template <typename Scalar>
AngularMomentumDataTpl<Scalar>::AngularMomentumDataTpl(
    const AngularMomentumResidualTpl<Scalar> *model)
    : Base(model->ndx1, model->nu, model->ndx2, 3) {}

} // namespace aligator
