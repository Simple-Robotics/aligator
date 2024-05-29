/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/continuous-centroidal-fwd.hpp"

namespace aligator {
namespace dynamics {

template <typename Scalar>
ContinuousCentroidalFwdDynamicsTpl<Scalar>::ContinuousCentroidalFwdDynamicsTpl(
    const Manifold &state, const double mass, const Vector3s &gravity,
    const ContactMap &contact_map, const int force_size)
    : Base(state, (int)contact_map.getSize() * force_size), space_(state),
      nk_(contact_map.getSize()), mass_(mass), gravity_(gravity),
      contact_map_(contact_map), force_size_(force_size) {
  if (space_.nx() != 9 + nu_) {
    ALIGATOR_DOMAIN_ERROR(fmt::format("State space should be of size: "
                                      "({}).",
                                      9 + nu_));
  }
}

template <typename Scalar>
void ContinuousCentroidalFwdDynamicsTpl<Scalar>::forward(
    const ConstVectorRef &x, const ConstVectorRef &u, BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.xdot_.template head<3>() = 1 / mass_ * x.template segment<3>(3);
  d.xdot_.template segment<3>(3) = mass_ * gravity_;
  d.xdot_.template segment<3>(6).setZero();
  for (std::size_t i = 0; i < nk_; i++) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
    if (contact_map_.getContactState(i)) {
      d.xdot_.template segment<3>(3) +=
          x.template segment<3>(9 + i * force_size_);
      d.xdot_[6] += (contact_map_.getContactPose(i)[1] - x[1]) *
                        x[9 + i * force_size_ + 2] -
                    (contact_map_.getContactPose(i)[2] - x[2]) *
                        x[9 + i * force_size_ + 1];
      d.xdot_[7] +=
          (contact_map_.getContactPose(i)[2] - x[2]) * x[9 + i * force_size_] -
          (contact_map_.getContactPose(i)[0] - x[0]) *
              x[9 + i * force_size_ + 2];
      d.xdot_[8] +=
          (contact_map_.getContactPose(i)[0] - x[0]) *
              x[9 + i * force_size_ + 1] -
          (contact_map_.getContactPose(i)[1] - x[1]) * x[9 + i * force_size_];
      if (force_size_ == 6) {
        d.xdot_.template segment<3>(6) +=
            x.template segment<3>(9 + i * force_size_ + 3);
      }
    }
#pragma GCC diagnostic pop
  }
  d.xdot_.tail(nu_) = u;
}

template <typename Scalar>
void ContinuousCentroidalFwdDynamicsTpl<Scalar>::dForward(
    const ConstVectorRef &x, const ConstVectorRef &, BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  d.Jx_.setZero();
  d.Jx_.template block<3, 3>(0, 3).setIdentity();
  d.Jx_.template block<3, 3>(0, 3) /= mass_;
  for (std::size_t i = 0; i < nk_; i++) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
    if (contact_map_.getContactState(i)) {
      d.Jx_(6, 1) -= x[9 + i * force_size_ + 2];
      d.Jx_(6, 2) += x[9 + i * force_size_ + 1];
      d.Jx_(7, 0) += x[9 + i * force_size_ + 2];
      d.Jx_(7, 2) -= x[9 + i * force_size_];
      d.Jx_(8, 0) -= x[9 + i * force_size_ + 1];
      d.Jx_(8, 1) += x[9 + i * force_size_];

      d.Jtemp_ << 0.0, -(contact_map_.getContactPose(i)[2] - x[2]),
          (contact_map_.getContactPose(i)[1] - x[1]),
          (contact_map_.getContactPose(i)[2] - x[2]), 0.0,
          -(contact_map_.getContactPose(i)[0] - x[0]),
          -(contact_map_.getContactPose(i)[1] - x[1]),
          (contact_map_.getContactPose(i)[0] - x[0]), 0.0;

      d.Jx_.template block<3, 3>(6, 9 + force_size_ * i) = d.Jtemp_;
      d.Jx_.template block<3, 3>(3, 9 + force_size_ * i).setIdentity();
      if (force_size_ == 6) {
        d.Jx_.template block<3, 3>(6, 9 + force_size_ * i + 3).setIdentity();
      }
    }
  }
  d.Ju_.setZero();
  d.Ju_.block(9, 0, nu_, nu_).setIdentity();
#pragma GCC diagnostic pop
}

template <typename Scalar>
shared_ptr<ContinuousDynamicsDataTpl<Scalar>>
ContinuousCentroidalFwdDynamicsTpl<Scalar>::createData() const {
  return allocate_shared_eigen_aligned<Data>(this);
}

template <typename Scalar>
ContinuousCentroidalFwdDataTpl<Scalar>::ContinuousCentroidalFwdDataTpl(
    const ContinuousCentroidalFwdDynamicsTpl<Scalar> *cont_dyn)
    : Base(9 + cont_dyn->nu(), cont_dyn->nu()) {
  Jtemp_.setZero();
}
} // namespace dynamics
} // namespace aligator
