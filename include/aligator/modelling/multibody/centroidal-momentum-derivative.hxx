#pragma once

#include "aligator/modelling/multibody/centroidal-momentum-derivative.hpp"

#include <pinocchio/algorithm/centroidal-derivatives.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/center-of-mass-derivatives.hpp>

namespace aligator {

template <typename Scalar>
void CentroidalMomentumDerivativeResidualTpl<Scalar>::evaluate(
    const ConstVectorRef &x, const ConstVectorRef &u, const ConstVectorRef &,
    BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;

  const auto q = x.segment(6, pin_model_.nq);
  const auto v = x.tail(pin_model_.nv);
  const auto a = u.tail(pin_model_.nv);

  d.hdot_ = pinocchio::computeCentroidalMomentumTimeVariation(pin_model_, pdata,
                                                              q, v, a);

  d.value_.template head<3>() = d.hdot_.linear();
  d.value_.template head<3>() += mass_ * gravity_;
  d.value_.template tail<3>() = d.hdot_.angular();
  for (std::size_t i = 0; i < contact_map_.getSize(); i++) {
    if (contact_map_.getContactState(i)) {
      long i_ = static_cast<long>(i);
      d.value_.template head<3>() += u.template segment<3>(i_ * 3);
      d.value_[3] +=
          (contact_map_.getContactPose(i)[1] - pdata.com[0][1]) *
              u[i_ * 3 + 2] -
          (contact_map_.getContactPose(i)[2] - pdata.com[0][2]) * u[i_ * 3 + 1];
      d.value_[4] +=
          (contact_map_.getContactPose(i)[2] - pdata.com[0][2]) * u[i_ * 3] -
          (contact_map_.getContactPose(i)[0] - pdata.com[0][0]) * u[i_ * 3 + 2];
      d.value_[5] +=
          (contact_map_.getContactPose(i)[0] - pdata.com[0][0]) *
              u[i_ * 3 + 1] -
          (contact_map_.getContactPose(i)[1] - pdata.com[0][1]) * u[i_ * 3];
    }
  }
}

template <typename Scalar>
void CentroidalMomentumDerivativeResidualTpl<Scalar>::computeJacobians(
    const ConstVectorRef &x, const ConstVectorRef &u, const ConstVectorRef &,
    BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;

  const auto q = x.segment(6, pin_model_.nq);
  const auto v = x.tail(pin_model_.nv);
  const auto a = u.tail(pin_model_.nv);

  pinocchio::centerOfMass(pin_model_, pdata, q, v);
  pinocchio::jacobianCenterOfMass(pin_model_, pdata, q);
  pinocchio::computeCentroidalDynamicsDerivatives(pin_model_, pdata, q, v, a,
                                                  d.dh_dq_, d.dhdot_dq_,
                                                  d.dhdot_dv_, d.dhdot_da_);

  d.Jx_.template topRows<6>().setZero();
  d.Jx_.block(0, 6, 6, pin_model_.nv) = d.dhdot_dq_;
  for (std::size_t i = 0; i < contact_map_.getSize(); i++) {
    long i_ = static_cast<long>(i);
    if (contact_map_.getContactState(i)) {
      d.Jtemp_ << 0, -u[i_ * 3 + 2], u[i_ * 3 + 1], u[i_ * 3 + 2], 0,
          -u[i_ * 3], -u[i_ * 3 + 1], u[i_ * 3], 0;
      d.Jx_.block(3, 6, 3, pin_model_.nv) += d.Jtemp_ * pdata.Jcom;
    }
  }
  d.Jx_.rightCols(pin_model_.nv) = d.dhdot_dv_;

  d.Ju_.setZero();
  for (std::size_t i = 0; i < contact_map_.getSize(); i++) {
    long i_ = static_cast<long>(i);
    if (contact_map_.getContactState(i)) {
      d.Jtemp_ << 0.0, -(contact_map_.getContactPose(i)[2] - pdata.com[0][2]),
          (contact_map_.getContactPose(i)[1] - pdata.com[0][1]),
          (contact_map_.getContactPose(i)[2] - pdata.com[0][2]), 0.0,
          -(contact_map_.getContactPose(i)[0] - pdata.com[0][0]),
          -(contact_map_.getContactPose(i)[1] - pdata.com[0][1]),
          (contact_map_.getContactPose(i)[0] - pdata.com[0][0]), 0.0;

      d.Ju_.template block<3, 3>(3, 3 * i_) = d.Jtemp_;
      d.Ju_.template block<3, 3>(0, 3 * i_).setIdentity();
    }
  }
  d.Ju_.rightCols(pin_model_.nv) = d.dhdot_da_;
}

template <typename Scalar>
CentroidalMomentumDerivativeDataTpl<Scalar>::
    CentroidalMomentumDerivativeDataTpl(
        const CentroidalMomentumDerivativeResidualTpl<Scalar> *model)
    : Base(model->ndx1, model->nu, model->ndx2, 6),
      pin_data_(model->pin_model_), dh_dq_(6, model->pin_model_.nv),
      dhdot_dq_(6, model->pin_model_.nv), dhdot_dv_(6, model->pin_model_.nv),
      dhdot_da_(6, model->pin_model_.nv) {
  dh_dq_.setZero();
  dhdot_dq_.setZero();
  dhdot_dv_.setZero();
  dhdot_da_.setZero();
}

} // namespace aligator
