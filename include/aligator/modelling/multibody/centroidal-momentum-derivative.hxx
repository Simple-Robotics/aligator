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

  const auto q = x.head(pin_model_.nq);
  const auto v = x.tail(pin_model_.nv);

  pinocchio::forwardKinematics(pin_model_, pdata, q);
  pinocchio::centerOfMass(pin_model_, pdata, q, v);

  d.value_.template head<3>() = mass_ * gravity_;
  d.value_.template tail<3>().setZero();
  for (std::size_t i = 0; i < contact_states_.size(); i++) {
    if (contact_states_[i]) {
      long i_ = static_cast<long>(i);
      d.value_.template head<3>() += u.template segment<3>(i_ * force_size_);
      pinocchio::updateFramePlacement(pin_model_, pdata, contact_ids_[i]);
      d.value_[3] +=
          (pdata.oMf[contact_ids_[i]].translation()[1] - pdata.com[0][1]) *
              u[i_ * force_size_ + 2] -
          (pdata.oMf[contact_ids_[i]].translation()[2] - pdata.com[0][2]) *
              u[i_ * force_size_ + 1];
      d.value_[4] +=
          (pdata.oMf[contact_ids_[i]].translation()[2] - pdata.com[0][2]) *
              u[i_ * force_size_] -
          (pdata.oMf[contact_ids_[i]].translation()[0] - pdata.com[0][0]) *
              u[i_ * force_size_ + 2];
      d.value_[5] +=
          (pdata.oMf[contact_ids_[i]].translation()[0] - pdata.com[0][0]) *
              u[i_ * force_size_ + 1] -
          (pdata.oMf[contact_ids_[i]].translation()[1] - pdata.com[0][1]) *
              u[i_ * force_size_];
      if (force_size_ == 6) {
        d.value_.template tail<3>() +=
            u.template segment<3>(i_ * force_size_ + 3);
      }
    }
  }
}

template <typename Scalar>
void CentroidalMomentumDerivativeResidualTpl<Scalar>::computeJacobians(
    const ConstVectorRef &x, const ConstVectorRef &u, const ConstVectorRef &,
    BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;

  const auto q = x.head(pin_model_.nq);

  pinocchio::jacobianCenterOfMass(pin_model_, pdata, q);
  pinocchio::computeJointJacobians(pin_model_, pdata);

  d.Jx_.setZero();
  for (std::size_t i = 0; i < contact_states_.size(); i++) {
    long i_ = static_cast<long>(i);
    if (contact_states_[i]) {
      d.fJf_.setZero();
      pinocchio::getFrameJacobian(pin_model_, pdata, contact_ids_[i],
                                  pinocchio::LOCAL_WORLD_ALIGNED, d.fJf_);
      d.Jtemp_ << 0, -u[i_ * force_size_ + 2], u[i_ * force_size_ + 1],
          u[i_ * force_size_ + 2], 0, -u[i_ * force_size_],
          -u[i_ * force_size_ + 1], u[i_ * force_size_], 0;
      d.Jx_.bottomLeftCorner(3, pin_model_.nv) +=
          d.Jtemp_ * (pdata.Jcom - d.fJf_.template topRows<3>());
    }
  }

  d.Ju_.setZero();
  for (std::size_t i = 0; i < contact_states_.size(); i++) {
    long i_ = static_cast<long>(i);
    if (contact_states_[i]) {
      d.Jtemp_ << 0.0,
          -(pdata.oMf[contact_ids_[i]].translation()[2] - pdata.com[0][2]),
          (pdata.oMf[contact_ids_[i]].translation()[1] - pdata.com[0][1]),
          (pdata.oMf[contact_ids_[i]].translation()[2] - pdata.com[0][2]), 0.0,
          -(pdata.oMf[contact_ids_[i]].translation()[0] - pdata.com[0][0]),
          -(pdata.oMf[contact_ids_[i]].translation()[1] - pdata.com[0][1]),
          (pdata.oMf[contact_ids_[i]].translation()[0] - pdata.com[0][0]), 0.0;

      d.Ju_.template block<3, 3>(0, force_size_ * i_).setIdentity();
      d.Ju_.template block<3, 3>(3, force_size_ * i_) = d.Jtemp_;
      if (force_size_ == 6) {
        d.Ju_.template block<3, 3>(3, force_size_ * i_ + 3).setIdentity();
      }
    }
  }
}

template <typename Scalar>
CentroidalMomentumDerivativeDataTpl<Scalar>::
    CentroidalMomentumDerivativeDataTpl(
        const CentroidalMomentumDerivativeResidualTpl<Scalar> *model)
    : Base(model->ndx1, model->nu, model->ndx2, 6),
      pin_data_(model->pin_model_), fJf_(6, model->pin_model_.nv) {
  fJf_.setZero();
}

} // namespace aligator
