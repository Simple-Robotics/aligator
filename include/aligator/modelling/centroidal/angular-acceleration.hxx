#pragma once

#include "aligator/modelling/centroidal/angular-acceleration.hpp"

namespace aligator {

template <typename Scalar>
void AngularAccelerationResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                      const ConstVectorRef &u,
                                                      const ConstVectorRef &,
                                                      BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.value_.setZero();
  for (std::size_t i = 0; i < nk_; i++) {
    long i_ = static_cast<long>(i);
    if (contact_map_.contact_states_[i]) {
      d.value_[0] +=
          (contact_map_.contact_poses_[i][1] - x[1]) * u[i_ * force_size_ + 2] -
          (contact_map_.contact_poses_[i][2] - x[2]) * u[i_ * force_size_ + 1];
      d.value_[1] +=
          (contact_map_.contact_poses_[i][2] - x[2]) * u[i_ * force_size_] -
          (contact_map_.contact_poses_[i][0] - x[0]) * u[i_ * force_size_ + 2];
      d.value_[2] +=
          (contact_map_.contact_poses_[i][0] - x[0]) * u[i_ * force_size_ + 1] -
          (contact_map_.contact_poses_[i][1] - x[1]) * u[i_ * force_size_];
      u[i_ * force_size_];
      if (force_size_ == 6) {
        d.value_.noalias() += u.template segment<3>(i_ * force_size_ + 3);
      }
    }
  }
}

template <typename Scalar>
void AngularAccelerationResidualTpl<Scalar>::computeJacobians(
    const ConstVectorRef &x, const ConstVectorRef &u, const ConstVectorRef &,
    BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.Jx_.setZero();
  d.Ju_.setZero();
  for (std::size_t i = 0; i < nk_; i++) {
    long i_ = static_cast<long>(i);
    if (contact_map_.contact_states_[i]) {
      d.Jx_(0, 1) -= u[i_ * force_size_ + 2];
      d.Jx_(0, 2) += u[i_ * force_size_ + 1];
      d.Jx_(1, 0) += u[i_ * force_size_ + 2];
      d.Jx_(1, 2) -= u[i_ * force_size_];
      d.Jx_(2, 0) -= u[i_ * force_size_ + 1];
      d.Jx_(2, 1) += u[i_ * force_size_];

      d.Jtemp_ << 0.0, -(contact_map_.contact_poses_[i][2] - x[2]),
          (contact_map_.contact_poses_[i][1] - x[1]),
          (contact_map_.contact_poses_[i][2] - x[2]), 0.0,
          -(contact_map_.contact_poses_[i][0] - x[0]),
          -(contact_map_.contact_poses_[i][1] - x[1]),
          (contact_map_.contact_poses_[i][0] - x[0]), 0.0;

      d.Ju_.template block<3, 3>(0, force_size_ * i_) = d.Jtemp_;

      if (force_size_ == 6) {
        d.Ju_.template block<3, 3>(0, force_size_ * i_ + 3).setIdentity();
      }
    }
  }
}

template <typename Scalar>
AngularAccelerationDataTpl<Scalar>::AngularAccelerationDataTpl(
    const AngularAccelerationResidualTpl<Scalar> *model)
    : Base(model->ndx1, model->nu, model->ndx2, 3) {
  Jtemp_.setZero();
}

} // namespace aligator
