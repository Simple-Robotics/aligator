/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/kinodynamics-fwd.hpp"

#include <pinocchio/algorithm/centroidal-derivatives.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/center-of-mass-derivatives.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>

namespace aligator {
namespace dynamics {

template <typename Scalar>
KinodynamicsFwdDynamicsTpl<Scalar>::KinodynamicsFwdDynamicsTpl(
    const ManifoldPtr &state, const Model &model, const Vector3s &gravity,
    const ContactMap &contact_map,
    const std::vector<pinocchio::FrameIndex> frame_ids)
    : Base(state, model.nv - 6 + int(contact_map.getSize()) * 3), space_(state),
      pin_model_(model), gravity_(gravity), contact_map_(contact_map),
      frame_ids_(frame_ids) {
  mass_ = pinocchio::computeTotalMass(pin_model_);
}

template <typename Scalar>
void KinodynamicsFwdDynamicsTpl<Scalar>::forward(const ConstVectorRef &x,
                                                 const ConstVectorRef &u,
                                                 BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;
  const auto q = x.head(pin_model_.nq);
  const auto v = x.tail(pin_model_.nv);
  const auto a = u.tail(pin_model_.nv - 6);

  pinocchio::ccrba(pin_model_, pdata, q, v);  // Compute Ag
  pinocchio::dccrba(pin_model_, pdata, q, v); // Compute Ag_dot

  pinocchio::forwardKinematics(pin_model_, pdata, q);
  pinocchio::centerOfMass(pin_model_, pdata, q, v);

  d.Agu_inv_ = pdata.Ag.leftCols(6).inverse();

  // Compute external forces component
  d.cforces_.setZero();
  d.cforces_.template head<3>() = mass_ * gravity_;
  for (std::size_t i = 0; i < contact_map_.getSize(); i++) {
    if (contact_map_.getContactState(i)) {
      long i_ = static_cast<long>(i);
      pinocchio::updateFramePlacement(pin_model_, pdata, frame_ids_[i]);
      d.cforces_.template head<3>() += u.template segment<3>(i_ * 3);
      d.cforces_[3] +=
          (pdata.oMf[frame_ids_[i]].translation()[1] - pdata.com[0][1]) *
              u[i_ * 3 + 2] -
          (pdata.oMf[frame_ids_[i]].translation()[2] - pdata.com[0][2]) *
              u[i_ * 3 + 1];
      d.cforces_[4] +=
          (pdata.oMf[frame_ids_[i]].translation()[2] - pdata.com[0][2]) *
              u[i_ * 3] -
          (pdata.oMf[frame_ids_[i]].translation()[0] - pdata.com[0][0]) *
              u[i_ * 3 + 2];
      d.cforces_[5] +=
          (pdata.oMf[frame_ids_[i]].translation()[0] - pdata.com[0][0]) *
              u[i_ * 3 + 1] -
          (pdata.oMf[frame_ids_[i]].translation()[1] - pdata.com[0][1]) *
              u[i_ * 3];
    }
  }

  // Compute base acceleration with respect to whole-body motion and centroidal
  // dynamics
  d.xdot_.segment(pin_model_.nv, 6) =
      d.Agu_inv_ *
      (d.cforces_ - pdata.dAg * v - pdata.Ag.rightCols(pin_model_.nv - 6) * a);

  // Simple kinematics integration
  d.xdot_.head(pin_model_.nv) = v;
  d.xdot_.tail(pin_model_.nv - 6) = a;
}

template <typename Scalar>
void KinodynamicsFwdDynamicsTpl<Scalar>::dForward(const ConstVectorRef &x,
                                                  const ConstVectorRef &u,
                                                  BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;
  const auto q = x.head(pin_model_.nq);
  const auto v = x.tail(pin_model_.nv);
  const auto a = u.tail(pin_model_.nv - 6);

  pinocchio::centerOfMass(pin_model_, pdata, q, v);
  pinocchio::jacobianCenterOfMass(pin_model_, pdata, q);

  pinocchio::computeJointJacobians(pin_model_, pdata);

  ////// Jx computation //////
  d.Jtemp_.setZero();
  d.Jx_.block(pin_model_.nv, 0, 6, pin_model_.nv).setZero();
  // Compute kinematics terms in centroidal dynamics
  for (std::size_t i = 0; i < contact_map_.getSize(); i++) {
    long i_ = static_cast<long>(i);
    if (contact_map_.getContactState(i)) {
      d.fJf_.setZero();
      pinocchio::getFrameJacobian(pin_model_, pdata, frame_ids_[i],
                                  pinocchio::LOCAL_WORLD_ALIGNED, d.fJf_);
      d.Jtemp_ << 0, -u[i_ * 3 + 2], u[i_ * 3 + 1], u[i_ * 3 + 2], 0,
          -u[i_ * 3], -u[i_ * 3 + 1], u[i_ * 3], 0;
      d.Jx_.block(pin_model_.nv, 0, 6, pin_model_.nv) +=
          d.Agu_inv_.template rightCols<3>() * d.Jtemp_ *
          (pdata.Jcom - d.fJf_.template topRows<3>());
    }
  }
  // Compute d(Ag * q_ddot)/ dq
  d.a0_.tail(pin_model_.nv - 6) = a;
  pinocchio::computeCentroidalDynamicsDerivatives(
      pin_model_, pdata, q, VectorXs::Zero(pin_model_.nv), d.a0_, d.dh_dq_,
      d.dhdot_dq_, d.dhdot_dv_, d.dhdot_da_);
  d.Jx_.block(pin_model_.nv, 0, 6, pin_model_.nv) -= d.Agu_inv_ * d.dhdot_dq_;

  // Compute d(Ag_dot * q_dot)/ dq
  pinocchio::computeCentroidalDynamicsDerivatives(
      pin_model_, pdata, q, v, VectorXs::Zero(pin_model_.nv), d.dh_dq_,
      d.dhdot_dq_, d.dhdot_dv_, d.dhdot_da_);
  d.Jx_.block(pin_model_.nv, 0, 6, pin_model_.nv) -= d.Agu_inv_ * d.dhdot_dq_;

  // Compute d(Ag_dot * q_dot)/ dq_dot
  d.Jx_.block(pin_model_.nv, pin_model_.nv, 6, pin_model_.nv) =
      -d.Agu_inv_ * d.dhdot_dv_;

  // Compute dAgu_inv / dq
  d.a0_.setZero();
  d.a0_.template head<6>() = d.xdot_.segment(pin_model_.nv, 6);
  pinocchio::computeCentroidalDynamicsDerivatives(
      pin_model_, pdata, q, VectorXs::Zero(pin_model_.nv), d.a0_, d.dh_dq_,
      d.dhdot_dq_, d.dhdot_dv_, d.dhdot_da_);

  d.Jx_.block(pin_model_.nv, 0, 6, pin_model_.nv) -= d.Agu_inv_ * d.dhdot_dq_;

  ////// Ju computation //////

  // Compute derivatives with respect to forces
  d.Ju_.block(pin_model_.nv, 0, 6, nu_).setZero();
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

      d.Ju_.block(pin_model_.nv, 3 * i_, 6, 3) =
          d.Agu_inv_.template leftCols<3>();
      d.Ju_.block(pin_model_.nv, 3 * i_, 6, 3) +=
          d.Agu_inv_.template rightCols<3>() * d.Jtemp_;
    }
  }

  // Compute derivatives with respect to joint acceleration
  d.Ju_.block(pin_model_.nv, contact_map_.getSize() * 3, 6, pin_model_.nv - 6) =
      -d.Agu_inv_ * pdata.Ag.rightCols(pin_model_.nv - 6);
}

template <typename Scalar>
shared_ptr<ContinuousDynamicsDataTpl<Scalar>>
KinodynamicsFwdDynamicsTpl<Scalar>::createData() const {
  return allocate_shared_eigen_aligned<Data>(this);
}

template <typename Scalar>
KinodynamicsFwdDataTpl<Scalar>::KinodynamicsFwdDataTpl(
    const KinodynamicsFwdDynamicsTpl<Scalar> *model)
    : Base(model->ndx(), model->nu()), pin_data_(model->pin_model_),
      dh_dq_(6, model->pin_model_.nv), dhdot_dq_(6, model->pin_model_.nv),
      dhdot_dv_(6, model->pin_model_.nv), dhdot_da_(6, model->pin_model_.nv),
      fJf_(6, model->pin_model_.nv), a0_(model->pin_model_.nv) {
  this->Jx_.topRightCorner(model->pin_model_.nv, model->pin_model_.nv)
      .setIdentity();
  this->Ju_
      .bottomRightCorner(model->pin_model_.nv - 6, model->pin_model_.nv - 6)
      .setIdentity();
  Jtemp_.setZero();
  fJf_.setZero();
  Agu_inv_.setZero();
  a0_.setZero();
}
} // namespace dynamics
} // namespace aligator
