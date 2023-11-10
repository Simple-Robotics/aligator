#pragma once

#include "proxddp/modelling/multibody/center-of-mass-velocity.hpp"
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/center-of-mass-derivatives.hpp>

namespace proxddp {

template <typename Scalar>
void CenterOfMassVelocityResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                       BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const Model &model = *pin_model_;
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;
  pinocchio::centerOfMass(model, pdata, x.head(model.nq),
                          x.segment(model.nq, model.nv));

  d.value_ = pdata.vcom[0] - v_ref_;
}

template <typename Scalar>
void CenterOfMassVelocityResidualTpl<Scalar>::computeJacobians(
    const ConstVectorRef &x, BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const Model &model = *pin_model_;
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;

  pinocchio::centerOfMass(model, pdata, x.head(model.nq),
                          x.segment(model.nq, model.nv));
  pinocchio::getCenterOfMassVelocityDerivatives(model, pdata, d.fJf_);
  d.Jx_.leftCols(model.nv) = d.fJf_;

  pinocchio::jacobianCenterOfMass(model, pdata, x.head(model.nq));
  d.Jx_.rightCols(model.nv) = pdata.Jcom;
}

template <typename Scalar>
CenterOfMassVelocityDataTpl<Scalar>::CenterOfMassVelocityDataTpl(
    const CenterOfMassVelocityResidualTpl<Scalar> &model)
    : Base(model.ndx1, model.nu, model.ndx2, 3), pin_data_(*model.pin_model_),
      fJf_(3, model.pin_model_->nv) {
  fJf_.setZero();
}

} // namespace proxddp
