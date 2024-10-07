#pragma once

#include "aligator/modelling/centroidal/angular-momentum.hpp"

namespace aligator {

template <typename Scalar>
void AngularMomentumResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                  BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.value_ = x.template segment<3>(6) - L_ref_;
}

template <typename Scalar>
void AngularMomentumResidualTpl<Scalar>::computeJacobians(
    const ConstVectorRef &, BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.Jx_.template rightCols<3>().setIdentity();
}

template <typename Scalar>
AngularMomentumDataTpl<Scalar>::AngularMomentumDataTpl(
    const AngularMomentumResidualTpl<Scalar> *model)
    : Base(*model) {}

} // namespace aligator
