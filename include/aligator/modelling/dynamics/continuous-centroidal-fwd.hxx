/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/continuous-centroidal-fwd.hpp"

namespace aligator {
namespace dynamics {

template <typename Scalar>
ContinuousCentroidalFwdDynamicsTpl<Scalar>::ContinuousCentroidalFwdDynamicsTpl(
    const ManifoldPtr &state, const double mass, const Vector3s &gravity,
    const std::vector<std::pair<bool, Vector3s>> &contact_map)
    : Base(state, contact_map.size() * 3), space_(state),
      nk_(contact_map.size()), mass_(mass), gravity_(gravity),
      contact_map_(contact_map) {
  if (space_->nx() != 9 + nu_) {
    ALIGATOR_DOMAIN_ERROR(fmt::format("State space should be of size: "
                                      "({}).",
                                      9 + nu_));
  }
}

template <typename Scalar>
void ContinuousCentroidalFwdDynamicsTpl<Scalar>::forward(
    const ConstVectorRef &x, const ConstVectorRef &u, BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.xdot_.head(3) = 1 / mass_ * x.segment(3, 3);
  d.xdot_.segment(3, 3) = mass_ * gravity_;
  d.xdot_.segment(6, 3).setZero();
  for (std::size_t i = 0; i < nk_; i++) {
    const auto &it = contact_map_[i];
    if (it.first) {
      d.xdot_.segment(3, 3) += x.segment(9 + i * 3, 3);
      d.xdot_[6] += (it.second[1] - x[1]) * x[9 + i * 3 + 2] -
                    (it.second[2] - x[2]) * x[9 + i * 3 + 1];
      d.xdot_[7] += (it.second[2] - x[2]) * x[9 + i * 3] -
                    (it.second[0] - x[0]) * x[9 + i * 3 + 2];
      d.xdot_[8] += (it.second[0] - x[0]) * x[9 + i * 3 + 1] -
                    (it.second[1] - x[1]) * x[9 + i * 3];
    }
  }
  d.xdot_.tail(nu_) = u;
}

template <typename Scalar>
void ContinuousCentroidalFwdDynamicsTpl<Scalar>::dForward(
    const ConstVectorRef &x, const ConstVectorRef &u, BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  d.Jx_.setZero();
  d.Jx_.block(0, 3, 3, 3) = 1 / mass_ * Matrix3s::Identity();
  for (std::size_t i = 0; i < nk_; i++) {
    const auto &it = contact_map_[i];
    if (it.first) {
      d.Jx_(6, 1) -= x[9 + i * 3 + 2];
      d.Jx_(6, 2) += x[9 + i * 3 + 1];
      d.Jx_(7, 0) += x[9 + i * 3 + 2];
      d.Jx_(7, 2) -= x[9 + i * 3];
      d.Jx_(8, 0) -= x[9 + i * 3 + 1];
      d.Jx_(8, 1) += x[9 + i * 3];

      d.Jx_.block(6, 9 + 3 * i, 3, 3) << 0.0, -(it.second[2] - x[2]),
          (it.second[1] - x[1]), (it.second[2] - x[2]), 0.0,
          -(it.second[0] - x[0]), -(it.second[1] - x[1]), (it.second[0] - x[0]),
          0.0;
      d.Jx_.block(3, 9 + 3 * i, 3, 3).setIdentity();
    }
  }
  d.Ju_.setZero();
  d.Ju_.block(9, 0, nu_, nu_).setIdentity();
}

template <typename Scalar>
shared_ptr<ContinuousDynamicsDataTpl<Scalar>>
ContinuousCentroidalFwdDynamicsTpl<Scalar>::createData() const {
  return allocate_shared_eigen_aligned<Data>(this);
}

template <typename Scalar>
ContinuousCentroidalFwdDataTpl<Scalar>::ContinuousCentroidalFwdDataTpl(
    const ContinuousCentroidalFwdDynamicsTpl<Scalar> *cont_dyn)
    : Base(9 + cont_dyn->nu(), cont_dyn->nu()) {}
} // namespace dynamics
} // namespace aligator
