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
    const std::vector<bool> &contact_states,
    const std::vector<pinocchio::FrameIndex> &contact_ids, const int force_size)
    : Base(state, model.nv - 6 + int(contact_states.size()) * force_size),
      space_(state), pin_model_(model), gravity_(gravity),
      force_size_(force_size), contact_states_(contact_states),
      contact_ids_(contact_ids) {
  mass_ = pinocchio::computeTotalMass(pin_model_);
  if (contact_ids_.size() != contact_states_.size()) {
    ALIGATOR_DOMAIN_ERROR(
        fmt::format("contact_ids and contact_states should have same size: "
                    "now ({} and {}).",
                    contact_ids_.size(), contact_states_.size()));
  }
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

  d.PivLU_.compute(pdata.Ag.leftCols(6));
  d.Agu_inv_ = d.PivLU_.inverse();

  // Compute external forces component
  d.cforces_.setZero();
  d.cforces_.template head<3>() = mass_ * gravity_;
  for (std::size_t i = 0; i < contact_states_.size(); i++) {
    if (contact_states_[i]) {
      long i_ = static_cast<long>(i);
      pinocchio::updateFramePlacement(pin_model_, pdata, contact_ids_[i]);
      d.cforces_.template head<3>() += u.template segment<3>(i_ * force_size_);
      d.cforces_[3] +=
          (pdata.oMf[contact_ids_[i]].translation()[1] - pdata.com[0][1]) *
              u[i_ * force_size_ + 2] -
          (pdata.oMf[contact_ids_[i]].translation()[2] - pdata.com[0][2]) *
              u[i_ * force_size_ + 1];
      d.cforces_[4] +=
          (pdata.oMf[contact_ids_[i]].translation()[2] - pdata.com[0][2]) *
              u[i_ * force_size_] -
          (pdata.oMf[contact_ids_[i]].translation()[0] - pdata.com[0][0]) *
              u[i_ * force_size_ + 2];
      d.cforces_[5] +=
          (pdata.oMf[contact_ids_[i]].translation()[0] - pdata.com[0][0]) *
              u[i_ * force_size_ + 1] -
          (pdata.oMf[contact_ids_[i]].translation()[1] - pdata.com[0][1]) *
              u[i_ * force_size_];
      if (force_size_ == 6) {
        d.cforces_.template tail<3>() +=
            u.template segment<3>(i_ * force_size_ + 3);
      }
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
  for (std::size_t i = 0; i < contact_states_.size(); i++) {
    long i_ = static_cast<long>(i);
    if (contact_states_[i]) {
      d.fJf_.setZero();
      pinocchio::getFrameJacobian(pin_model_, pdata, contact_ids_[i],
                                  pinocchio::LOCAL_WORLD_ALIGNED, d.fJf_);
      d.Jtemp_ << 0, -u[i_ * force_size_ + 2], u[i_ * force_size_ + 1],
          u[i_ * force_size_ + 2], 0, -u[i_ * force_size_],
          -u[i_ * force_size_ + 1], u[i_ * force_size_], 0;
      d.temp1_.noalias() = d.Agu_inv_.template rightCols<3>() * d.Jtemp_;
      d.temp2_.noalias() = pdata.Jcom - d.fJf_.template topRows<3>();
      d.Jx_.block(pin_model_.nv, 0, 6, pin_model_.nv).noalias() +=
          d.temp1_ * d.temp2_;
    }
  }
  // Compute d(Ag * q_ddot)/ dq
  d.a0_.template head<6>().setZero();
  d.a0_.tail(pin_model_.nv - 6) = a;
  pinocchio::computeCentroidalDynamicsDerivatives(pin_model_, pdata, q, d.v0_,
                                                  d.a0_, d.dh_dq_, d.dhdot_dq_,
                                                  d.dhdot_dv_, d.dhdot_da_);
  d.Jx_.block(pin_model_.nv, 0, 6, pin_model_.nv).noalias() -=
      d.Agu_inv_ * d.dhdot_dq_;

  // Compute d(Ag_dot * q_dot)/ dq
  d.a0_.setZero();
  pinocchio::computeCentroidalDynamicsDerivatives(pin_model_, pdata, q, v,
                                                  d.a0_, d.dh_dq_, d.dhdot_dq_,
                                                  d.dhdot_dv_, d.dhdot_da_);

  d.Jx_.block(pin_model_.nv, 0, 6, pin_model_.nv).noalias() -=
      d.Agu_inv_ * d.dhdot_dq_;

  // Compute d(Ag_dot * q_dot)/ dq_dot
  d.Jx_.block(pin_model_.nv, pin_model_.nv, 6, pin_model_.nv).noalias() =
      -d.Agu_inv_ * d.dhdot_dv_;

  // Compute dAgu_inv / dq
  d.a0_.setZero();
  d.a0_.template head<6>() = d.xdot_.segment(pin_model_.nv, 6);

  pinocchio::computeCentroidalDynamicsDerivatives(
      pin_model_, pdata, q, VectorXs::Zero(pin_model_.nv), d.a0_, d.dh_dq_,
      d.dhdot_dq_, d.dhdot_dv_, d.dhdot_da_);

  d.Jx_.block(pin_model_.nv, 0, 6, pin_model_.nv).noalias() -=
      d.Agu_inv_ * d.dhdot_dq_;

  ////// Ju computation //////

  // Compute derivatives with respect to forces
  d.Ju_.block(pin_model_.nv, 0, 6, nu_).setZero();
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

      d.Ju_.block(pin_model_.nv, force_size_ * i_, 6, 3).noalias() =
          d.Agu_inv_.template leftCols<3>();
      d.Ju_.block(pin_model_.nv, force_size_ * i_, 6, 3).noalias() +=
          d.Agu_inv_.template rightCols<3>() * d.Jtemp_;
      if (force_size_ == 6) {
        d.Ju_.block(pin_model_.nv, force_size_ * i_ + 3, 6, 3).noalias() +=
            d.Agu_inv_.template rightCols<3>();
      }
    }
  }

  // Compute derivatives with respect to joint acceleration
  d.Ju_
      .block(pin_model_.nv, (long)contact_states_.size() * force_size_, 6,
             pin_model_.nv - 6)
      .noalias() = -d.Agu_inv_ * pdata.Ag.rightCols(pin_model_.nv - 6);
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
      temp1_(6, 3), temp2_(3, model->pin_model_.nv),
      fJf_(6, model->pin_model_.nv), v0_(model->pin_model_.nv),
      a0_(model->pin_model_.nv), PivLU_(6) {
  this->Jx_.topRightCorner(model->pin_model_.nv, model->pin_model_.nv)
      .setIdentity();
  this->Ju_
      .bottomRightCorner(model->pin_model_.nv - 6, model->pin_model_.nv - 6)
      .setIdentity();

  dh_dq_.setZero();
  dhdot_dq_.setZero();
  dhdot_dv_.setZero();
  dhdot_da_.setZero();
  temp1_.setZero();
  temp2_.setZero();
  fJf_.setZero();
  v0_.setZero();
  a0_.setZero();
  cforces_.setZero();
  Jtemp_.setZero();
  Agu_inv_.setZero();
}
} // namespace dynamics
} // namespace aligator
