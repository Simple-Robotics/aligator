/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/centroidal-kinematics-fwd.hpp"

#include <pinocchio/algorithm/centroidal-derivatives.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/center-of-mass-derivatives.hpp>

namespace aligator {
namespace dynamics {

template <typename Scalar>
CentroidalKinematicsFwdDynamicsTpl<Scalar>::CentroidalKinematicsFwdDynamicsTpl(
    const ManifoldPtr &state, const Model &model, const Vector3s &gravity,
    const ContactMap &contact_map)
    : Base(state, model.nv + int(contact_map.getSize()) * 3), space_(state),
      pin_model_(model), gravity_(gravity), contact_map_(contact_map) {
  mass_ = pinocchio::computeTotalMass(pin_model_);
}

template <typename Scalar>
void CentroidalKinematicsFwdDynamicsTpl<Scalar>::forward(
    const ConstVectorRef &x, const ConstVectorRef &u, BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;
  const auto q = x.segment(6, pin_model_.nq);
  const auto v = x.tail(pin_model_.nv);
  const auto a = u.tail(pin_model_.nv);

  d.hdot_ = pinocchio::computeCentroidalMomentumTimeVariation(pin_model_, pdata,
                                                              q, v, a);
  pinocchio::centerOfMass(pin_model_, pdata, q, v);

  d.xdot_.template head<3>() = d.hdot_.linear();
  d.xdot_.template head<3>() += mass_ * gravity_;
  d.xdot_.template segment<3>(3) = d.hdot_.angular();
  for (std::size_t i = 0; i < contact_map_.getSize(); i++) {
    if (contact_map_.getContactState(i)) {
      long i_ = static_cast<long>(i);
      d.xdot_.template head<3>() += u.template segment<3>(i_ * 3);
      d.xdot_[3] +=
          (contact_map_.getContactPose(i)[1] - pdata.com[0][1]) *
              u[i_ * 3 + 2] -
          (contact_map_.getContactPose(i)[2] - pdata.com[0][2]) * u[i_ * 3 + 1];
      d.xdot_[4] +=
          (contact_map_.getContactPose(i)[2] - pdata.com[0][2]) * u[i_ * 3] -
          (contact_map_.getContactPose(i)[0] - pdata.com[0][0]) * u[i_ * 3 + 2];
      d.xdot_[5] +=
          (contact_map_.getContactPose(i)[0] - pdata.com[0][0]) *
              u[i_ * 3 + 1] -
          (contact_map_.getContactPose(i)[1] - pdata.com[0][1]) * u[i_ * 3];
    }
  }
  d.xdot_.segment(9, pin_model_.nv) = v;
  d.xdot_.tail(pin_model_.nv) = a;
}

template <typename Scalar>
void CentroidalKinematicsFwdDynamicsTpl<Scalar>::dForward(
    const ConstVectorRef &x, const ConstVectorRef &u, BaseData &data) const {
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
          -u[i_ * 3], u[i_ * 3 + 1], -u[i_ * 3], 0;
      d.Jx_.block(3, 6, 3, pin_model_.nv) += d.Jtemp_ * pdata.Jcom;
    }
  }
  d.Jx_.block(0, 6 + pin_model_.nv, 6, pin_model_.nv) = d.dhdot_dv_;

  d.Ju_.template topRows<6>().setZero();
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
  d.Ju_.topRightCorner(6, pin_model_.nv) = d.dhdot_da_;
}

template <typename Scalar>
shared_ptr<ContinuousDynamicsDataTpl<Scalar>>
CentroidalKinematicsFwdDynamicsTpl<Scalar>::createData() const {
  return allocate_shared_eigen_aligned<Data>(this);
}

template <typename Scalar>
CentroidalKinematicsFwdDataTpl<Scalar>::CentroidalKinematicsFwdDataTpl(
    const CentroidalKinematicsFwdDynamicsTpl<Scalar> *model)
    : Base(model->ndx(), model->nu()), pin_data_(model->pin_model_),
      dh_dq_(6, model->pin_model_.nv), dhdot_dq_(6, model->pin_model_.nv),
      dhdot_dv_(6, model->pin_model_.nv), dhdot_da_(6, model->pin_model_.nv) {
  this->Jx_
      .block(6, 6 + model->pin_model_.nv, model->pin_model_.nv,
             model->pin_model_.nv)
      .setIdentity();
  this->Ju_.bottomRightCorner(model->pin_model_.nv, model->pin_model_.nv)
      .setIdentity();
  Jtemp_.setZero();
}
} // namespace dynamics
} // namespace aligator
