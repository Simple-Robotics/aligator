/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/centroidal-fwd.hpp"

namespace aligator {
namespace dynamics {

template <typename Scalar>
CentroidalFwdDynamicsTpl<Scalar>::CentroidalFwdDynamicsTpl(
    const ManifoldPtr &state, const int &nk, const double &mass)
    : Base(state, nk * 6), space_(state), nk_(nk), mass_(mass) {
  gravity_ << 0, 0, -mass_ * 9.81;
  contact_points_ = std::vector<Vector3s>(nk_, Vector3s::Zero());
  active_contacts_ = std::vector<bool>(nk_, true);
}

template <typename Scalar>
void CentroidalFwdDynamicsTpl<Scalar>::forward(const ConstVectorRef &x,
                                               const ConstVectorRef &u,
                                               BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  d.xdot_.head(3) = 1 / mass_ * x.segment(3, 6);
  d.xdot_.segment(3, 6) = gravity_;
  d.xdot_.segment(6, 9) << 0, 0, 0;
  for (std::size_t i = 0; i < nk_; i++) {
    if (active_contacts_[i]) {
      d.xdot_.segment(3, 6) += u.segment(i * 6, i * 6 + 3);
      d.xdot_.segment(6, 9) += u.segment(i * 6 + 3, i * 6 + 6);
      d.xdot_[6] += (contact_points_[i][1] - x[1]) * u[i * 6 + 2] -
                    (contact_points_[i][2] - x[2]) * u[i * 6 + 1];
      d.xdot_[7] += (contact_points_[i][2] - x[2]) * u[i * 6] -
                    (contact_points_[i][0] - x[0]) * u[i * 6 + 2];
      d.xdot_[8] += (contact_points_[i][0] - x[0]) * u[i * 6 + 1] -
                    (contact_points_[i][1] - x[1]) * u[i * 6];
    }
  }
}

template <typename Scalar>
void CentroidalFwdDynamicsTpl<Scalar>::dForward(const ConstVectorRef &x,
                                                const ConstVectorRef &u,
                                                BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  d.Jx_.block(0, 3, 3, 3) = 1 / mass_ * Matrix3s::Identity();
  d.B.setZero();
  for (std::size_t i = 0; i < nk_; i++) {
    if (active_contacts_[i]) {
      d.B[0, 1] -= u[i * 6 + 2];
      d.B[0, 2] += u[i * 6 + 1];
      d.B[1, 0] += u[i * 6 + 2];
      d.B[1, 2] -= u[i * 6];
      d.B[2, 0] -= u[i * 6 + 1];
      d.B[2, 1] += u[i * 6];
    }
  }
  d.Jx_.bottomLeftCorner(3, 3) = d.B;
  d.Ju_.setZero();
  for (std::size_t i = 0; i < nk_; i++) {
    if (active_contacts_[i]) {
      d.Ju_.block(6, 6 * nk_, 3, 3) << 0.0, -(contact_points_[i][2] - x[2]),
          (contact_points_[i][1] - x[1]), (contact_points_[i][2] - x[2]), 0.0,
          -(contact_points_[i][0] - x[0]), -(contact_points_[i][1] - x[1]),
          (contact_points_[i][0] - x[0]), 0.0;
      d.Ju_.block(3, 6 * nk_, 3, 3) = Matrix3s::Identity();
      d.Ju_.block(6, 6 * nk_ + 3, 3, 3) = Matrix3s::Identity();
    }
  }
}

template <typename Scalar>
void CentroidalFwdDynamicsTpl<Scalar>::updateContactPoints(
    std::vector<Vector3s> contact_points) {
  if (contact_points.size() != nk_) {
    ALIGATOR_DOMAIN_ERROR(fmt::format(
        "contact points vector does not have the right size, should be {}.",
        nk_));
  }
  contact_points_ = contact_points;
}

template <typename Scalar>
void CentroidalFwdDynamicsTpl<Scalar>::updateGait(
    std::vector<bool> active_contacts) {
  if (active_contacts.size() != nk_) {
    ALIGATOR_DOMAIN_ERROR(fmt::format(
        "active contacts vector does not have the right size, should be {}.",
        nk_));
  }
  active_contacts_ = active_contacts;
}

template <typename Scalar>
shared_ptr<ContinuousDynamicsDataTpl<Scalar>>
CentroidalFwdDynamicsTpl<Scalar>::createData() const {
  return allocate_shared_eigen_aligned<Data>(this);
}

template <typename Scalar>
CentroidalFwdDataTpl<Scalar>::CentroidalFwdDataTpl(
    const CentroidalFwdDynamicsTpl<Scalar> *cont_dyn)
    : Base(9, 6 * cont_dyn->nk_), Fx_(9, 9), Fu_(9, 6 * cont_dyn->nk_),
      B(3, 3) {}
} // namespace dynamics
} // namespace aligator
