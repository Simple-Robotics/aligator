/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/centroidal-fwd.hpp"

namespace aligator {
namespace dynamics {

template <typename Scalar>
CentroidalFwdDynamicsTpl<Scalar>::CentroidalFwdDynamicsTpl(
    const Manifold &state, const double mass, const Vector3s &gravity,
    const ContactMap &contact_map, const int force_size)
    : Base(state, int(contact_map.size_) * force_size), space_(state),
      nk_(contact_map.size_), mass_(mass), gravity_(gravity),
      contact_map_(contact_map), force_size_(force_size) {}

template <typename Scalar>
void CentroidalFwdDynamicsTpl<Scalar>::forward(const ConstVectorRef &x,
                                               const ConstVectorRef &u,
                                               BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.xdot_.template head<3>() = 1 / mass_ * x.template segment<3>(3);
  d.xdot_.template segment<3>(3) = mass_ * gravity_;
  d.xdot_.template segment<3>(6).setZero();
  for (std::size_t i = 0; i < nk_; i++) {
    if (contact_map_.contact_states_[i]) {
      long i_ = static_cast<long>(i);
      d.xdot_.template segment<3>(3) += u.template segment<3>(i_ * force_size_);
      d.xdot_[6] +=
          (contact_map_.contact_poses_[i][1] - x[1]) * u[i_ * force_size_ + 2] -
          (contact_map_.contact_poses_[i][2] - x[2]) * u[i_ * force_size_ + 1];
      d.xdot_[7] +=
          (contact_map_.contact_poses_[i][2] - x[2]) * u[i_ * force_size_] -
          (contact_map_.contact_poses_[i][0] - x[0]) * u[i_ * force_size_ + 2];
      d.xdot_[8] +=
          (contact_map_.contact_poses_[i][0] - x[0]) * u[i_ * force_size_ + 1] -
          (contact_map_.contact_poses_[i][1] - x[1]) * u[i_ * force_size_];
      if (force_size_ == 6) {
        d.xdot_.template tail<3>() +=
            u.template segment<3>(i_ * force_size_ + 3);
      }
    }
  }
}

template <typename Scalar>
void CentroidalFwdDynamicsTpl<Scalar>::dForward(const ConstVectorRef &x,
                                                const ConstVectorRef &u,
                                                BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  d.Jx_.setZero();
  d.Ju_.setZero();
  d.Jx_.template block<3, 3>(0, 3).setIdentity();
  d.Jx_.template block<3, 3>(0, 3) /= mass_;
  for (std::size_t i = 0; i < nk_; i++) {
    long i_ = static_cast<long>(i);
    if (contact_map_.contact_states_[i]) {
      d.Jx_(6, 1) -= u[i_ * force_size_ + 2];
      d.Jx_(6, 2) += u[i_ * force_size_ + 1];
      d.Jx_(7, 0) += u[i_ * force_size_ + 2];
      d.Jx_(7, 2) -= u[i_ * force_size_];
      d.Jx_(8, 0) -= u[i_ * force_size_ + 1];
      d.Jx_(8, 1) += u[i_ * force_size_];

      d.Jtemp_ << 0.0, -(contact_map_.contact_poses_[i][2] - x[2]),
          (contact_map_.contact_poses_[i][1] - x[1]),
          (contact_map_.contact_poses_[i][2] - x[2]), 0.0,
          -(contact_map_.contact_poses_[i][0] - x[0]),
          -(contact_map_.contact_poses_[i][1] - x[1]),
          (contact_map_.contact_poses_[i][0] - x[0]), 0.0;

      d.Ju_.template block<3, 3>(6, force_size_ * i_) = d.Jtemp_;
      d.Ju_.template block<3, 3>(3, force_size_ * i_).setIdentity();
      if (force_size_ == 6) {
        d.Ju_.template block<3, 3>(6, force_size_ * i_ + 3).setIdentity();
      }
    }
  }
}

template <typename Scalar>
shared_ptr<ContinuousDynamicsDataTpl<Scalar>>
CentroidalFwdDynamicsTpl<Scalar>::createData() const {
  return std::make_shared<Data>(this);
}

template <typename Scalar>
CentroidalFwdDataTpl<Scalar>::CentroidalFwdDataTpl(
    const CentroidalFwdDynamicsTpl<Scalar> *cont_dyn)
    : Base(9, cont_dyn->force_size_ * int(cont_dyn->nk_)) {
  Jtemp_.setZero();
}
} // namespace dynamics
} // namespace aligator
