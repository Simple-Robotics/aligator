#pragma once

#include "aligator/modelling/multibody/dcm-position.hpp"
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/center-of-mass-derivatives.hpp>

namespace aligator {

template <typename Scalar>
void DCMPositionResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                              BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;
  pinocchio::centerOfMass(pin_model_, pdata, x.head(pin_model_.nq),
                          x.segment(pin_model_.nq, pin_model_.nv));

  d.value_ = pdata.com[0] + alpha_ * pdata.vcom[0] - dcm_ref_;
}

template <typename Scalar>
void DCMPositionResidualTpl<Scalar>::computeJacobians(const ConstVectorRef &x,
                                                      BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;

  pinocchio::getCenterOfMassVelocityDerivatives(pin_model_, pdata, d.fJf_);
  pinocchio::jacobianCenterOfMass(pin_model_, pdata, x.head(pin_model_.nq));

  d.Jx_.leftCols(pin_model_.nv) = pdata.Jcom + alpha_ * d.fJf_;
  d.Jx_.rightCols(pin_model_.nv) = alpha_ * pdata.Jcom;
}

template <typename Scalar>
DCMPositionDataTpl<Scalar>::DCMPositionDataTpl(
    const DCMPositionResidualTpl<Scalar> &model)
    : Base(model.ndx1, model.nu, 3)
    , pin_data_(model.pin_model_)
    , fJf_(3, model.pin_model_.nv) {
  fJf_.setZero();
}

} // namespace aligator
