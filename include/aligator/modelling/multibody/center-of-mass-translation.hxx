#pragma once

#include "aligator/modelling/multibody/center-of-mass-translation.hpp"
#include <pinocchio/algorithm/center-of-mass.hpp>

namespace aligator {

template <typename Scalar>
void CenterOfMassTranslationResidualTpl<Scalar>::evaluate(
    const ConstVectorRef &x, BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;
  pinocchio::centerOfMass(pin_model_, pdata, x.head(pin_model_.nq));

  d.value_ = pdata.com[0] - p_ref_;
}

template <typename Scalar>
void CenterOfMassTranslationResidualTpl<Scalar>::computeJacobians(
    const ConstVectorRef &x, BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;
  pinocchio::jacobianCenterOfMass(pin_model_, pdata, x.head(pin_model_.nq));

  d.Jx_.leftCols(pin_model_.nv) = pdata.Jcom;
}

template <typename Scalar>
CenterOfMassTranslationDataTpl<Scalar>::CenterOfMassTranslationDataTpl(
    const CenterOfMassTranslationResidualTpl<Scalar> *model)
    : Base(model->ndx1, model->nu, 3)
    , pin_data_(model->pin_model_) {}

} // namespace aligator
