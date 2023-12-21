#pragma once

#include "proxddp/modelling/multibody/center-of-mass-translation.hpp"
#include <pinocchio/algorithm/center-of-mass.hpp>

namespace aligator {

template <typename Scalar>
void CenterOfMassTranslationResidualTpl<Scalar>::evaluate(
    const ConstVectorRef &x, BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const Model &model = *pin_model_;
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;
  pinocchio::centerOfMass(model, pdata, x.head(model.nq));

  d.value_ = pdata.com[0] - p_ref_;
}

template <typename Scalar>
void CenterOfMassTranslationResidualTpl<Scalar>::computeJacobians(
    const ConstVectorRef &x, BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const Model &model = *pin_model_;
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;
  pinocchio::jacobianCenterOfMass(model, pdata, x.head(model.nq));

  d.Jx_.leftCols(model.nv) = pdata.Jcom;
}

template <typename Scalar>
CenterOfMassTranslationDataTpl<Scalar>::CenterOfMassTranslationDataTpl(
    const CenterOfMassTranslationResidualTpl<Scalar> *model)
    : Base(model->ndx1, model->nu, model->ndx2, 3),
      pin_data_(*model->pin_model_) {}

} // namespace aligator
