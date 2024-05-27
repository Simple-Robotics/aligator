#pragma once

#include "aligator/modelling/multibody/center-of-mass-velocity.hpp"
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/center-of-mass-derivatives.hpp>

namespace aligator {

template <typename Scalar>
void CenterOfMassVelocityResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                       BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;
  pinocchio::centerOfMass(pin_model_, pdata, x.head(pin_model_.nq),
                          x.segment(pin_model_.nq, pin_model_.nv));

  d.value_ = pdata.vcom[0] - v_ref_;
}

template <typename Scalar>
void CenterOfMassVelocityResidualTpl<Scalar>::computeJacobians(
    const ConstVectorRef &x, BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;

  pinocchio::centerOfMass(pin_model_, pdata, x.head(pin_model_.nq),
                          x.segment(pin_model_.nq, pin_model_.nv));
  pinocchio::getCenterOfMassVelocityDerivatives(pin_model_, pdata, d.fJf_);
  d.Jx_.leftCols(pin_model_.nv) = d.fJf_;

  pinocchio::jacobianCenterOfMass(pin_model_, pdata, x.head(pin_model_.nq));
  d.Jx_.rightCols(pin_model_.nv) = pdata.Jcom;
}

template <typename Scalar>
CenterOfMassVelocityDataTpl<Scalar>::CenterOfMassVelocityDataTpl(
    const CenterOfMassVelocityResidualTpl<Scalar> &model)
    : Base(model.ndx1, model.nu, model.ndx2, 3), pin_data_(model.pin_model_),
      fJf_(3, model.pin_model_.nv) {
  fJf_.setZero();
}

} // namespace aligator
