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
  for (std::size_t i = 0; i < contact_map_.getSize(); i++) {
    if (contact_map_.getContactState(i)) {
      long i_ = static_cast<long>(i);
      d.value_.template head<3>() += u.template segment<3>(i_ * 3);
      pinocchio::updateFramePlacement(pin_model_, pdata, frame_ids_[i]);
      d.value_[3] +=
          (pdata.oMf[frame_ids_[i]].translation()[1] - pdata.com[0][1]) *
              u[i_ * 3 + 2] -
          (pdata.oMf[frame_ids_[i]].translation()[2] - pdata.com[0][2]) *
              u[i_ * 3 + 1];
      d.value_[4] +=
          (pdata.oMf[frame_ids_[i]].translation()[2] - pdata.com[0][2]) *
              u[i_ * 3] -
          (pdata.oMf[frame_ids_[i]].translation()[0] - pdata.com[0][0]) *
              u[i_ * 3 + 2];
      d.value_[5] +=
          (pdata.oMf[frame_ids_[i]].translation()[0] - pdata.com[0][0]) *
              u[i_ * 3 + 1] -
          (pdata.oMf[frame_ids_[i]].translation()[1] - pdata.com[0][1]) *
              u[i_ * 3];
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
  const auto v = x.tail(pin_model_.nv);

  pinocchio::jacobianCenterOfMass(pin_model_, pdata, q);
  pinocchio::computeJointJacobians(pin_model_, pdata);

  d.Jx_.setZero();
  for (std::size_t i = 0; i < contact_map_.getSize(); i++) {
    long i_ = static_cast<long>(i);
    if (contact_map_.getContactState(i)) {
      d.fJf_.setZero();
      pinocchio::getFrameJacobian(pin_model_, pdata, frame_ids_[i],
                                  pinocchio::LOCAL_WORLD_ALIGNED, d.fJf_);
      d.Jtemp_ << 0, -u[i_ * 3 + 2], u[i_ * 3 + 1], u[i_ * 3 + 2], 0,
          -u[i_ * 3], -u[i_ * 3 + 1], u[i_ * 3], 0;
      d.Jx_.bottomLeftCorner(3, pin_model_.nv) +=
          d.Jtemp_ * (pdata.Jcom - d.fJf_.template topRows<3>());
    }
  }

  d.Ju_.setZero();
  for (std::size_t i = 0; i < contact_map_.getSize(); i++) {
    long i_ = static_cast<long>(i);
    if (contact_map_.getContactState(i)) {
      d.Jtemp_ << 0.0,
          -(pdata.oMf[frame_ids_[i]].translation()[2] - pdata.com[0][2]),
          (pdata.oMf[frame_ids_[i]].translation()[1] - pdata.com[0][1]),
          (pdata.oMf[frame_ids_[i]].translation()[2] - pdata.com[0][2]), 0.0,
          -(pdata.oMf[frame_ids_[i]].translation()[0] - pdata.com[0][0]),
          -(pdata.oMf[frame_ids_[i]].translation()[1] - pdata.com[0][1]),
          (pdata.oMf[frame_ids_[i]].translation()[0] - pdata.com[0][0]), 0.0;

      d.Ju_.template block<3, 3>(0, 3 * i_).setIdentity();
      d.Ju_.template block<3, 3>(3, 3 * i_) = d.Jtemp_;
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
