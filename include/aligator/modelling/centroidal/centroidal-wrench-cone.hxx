#pragma once

#include "aligator/modelling/centroidal/centroidal-wrench-cone.hpp"

namespace aligator {

template <typename Scalar>
void CentroidalWrenchConeResidualTpl<Scalar>::evaluate(const ConstVectorRef &,
                                                       const ConstVectorRef &u,
                                                       BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  // Unilateral contact
  d.value_[0] = -u[k_ * 6 + 2];

  // Coulomb friction inequalities
  d.value_[1] = -u[k_ * 6] - mu_ * u[k_ * 6 + 2];
  d.value_[2] = +u[k_ * 6] - mu_ * u[k_ * 6 + 2];
  d.value_[3] = -u[k_ * 6 + 1] - mu_ * u[k_ * 6 + 2];
  d.value_[4] = +u[k_ * 6 + 1] - mu_ * u[k_ * 6 + 2];

  // Local CoP inequalities
  d.value_[5] = -hW_ * u[k_ * 6 + 2] - u[k_ * 6 + 3];
  d.value_[6] = -hW_ * u[k_ * 6 + 2] + u[k_ * 6 + 3];
  d.value_[7] = -hL_ * u[k_ * 6 + 2] - u[k_ * 6 + 4];
  d.value_[8] = -hL_ * u[k_ * 6 + 2] + u[k_ * 6 + 4];

  // z-torque limits
  d.value_[9] = -hW_ * u[k_ * 6] - hL_ * u[k_ * 6 + 1] -
                (hW_ + hL_) * mu_ * u[k_ * 6 + 2] + mu_ * u[k_ * 6 + 3] +
                mu_ * u[k_ * 6 + 4] - u[k_ * 6 + 5];
  d.value_[10] = -hW_ * u[k_ * 6] + hL_ * u[k_ * 6 + 1] -
                 (hW_ + hL_) * mu_ * u[k_ * 6 + 2] + mu_ * u[k_ * 6 + 3] -
                 mu_ * u[k_ * 6 + 4] - u[k_ * 6 + 5];
  d.value_[11] = hW_ * u[k_ * 6] - hL_ * u[k_ * 6 + 1] -
                 (hW_ + hL_) * mu_ * u[k_ * 6 + 2] - mu_ * u[k_ * 6 + 3] +
                 mu_ * u[k_ * 6 + 4] - u[k_ * 6 + 5];
  d.value_[12] = hW_ * u[k_ * 6] + hL_ * u[k_ * 6 + 1] -
                 (hW_ + hL_) * mu_ * u[k_ * 6 + 2] - mu_ * u[k_ * 6 + 3] -
                 mu_ * u[k_ * 6 + 4] - u[k_ * 6 + 5];

  d.value_[13] = hW_ * u[k_ * 6] + hL_ * u[k_ * 6 + 1] -
                 (hW_ + hL_) * mu_ * u[k_ * 6 + 2] + mu_ * u[k_ * 6 + 3] +
                 mu_ * u[k_ * 6 + 4] + u[k_ * 6 + 5];
  d.value_[14] = hW_ * u[k_ * 6] - hL_ * u[k_ * 6 + 1] -
                 (hW_ + hL_) * mu_ * u[k_ * 6 + 2] + mu_ * u[k_ * 6 + 3] -
                 mu_ * u[k_ * 6 + 4] + u[k_ * 6 + 5];
  d.value_[15] = -hW_ * u[k_ * 6] + hL_ * u[k_ * 6 + 1] -
                 (hW_ + hL_) * mu_ * u[k_ * 6 + 2] - mu_ * u[k_ * 6 + 3] +
                 mu_ * u[k_ * 6 + 4] + u[k_ * 6 + 5];
  d.value_[16] = -hW_ * u[k_ * 6] - hL_ * u[k_ * 6 + 1] -
                 (hW_ + hL_) * mu_ * u[k_ * 6 + 2] - mu_ * u[k_ * 6 + 3] -
                 mu_ * u[k_ * 6 + 4] + u[k_ * 6 + 5];
}

template <typename Scalar>
void CentroidalWrenchConeResidualTpl<Scalar>::computeJacobians(
    const ConstVectorRef &, const ConstVectorRef &, BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.Jtemp_ << 0, 0, -1, 0, 0, 0, -1, 0, -mu_, 0, 0, 0, 1, 0, -mu_, 0, 0, 0, 0,
      -1, -mu_, 0, 0, 0, 0, 1, -mu_, 0, 0, 0, 0, 0, -hW_, -1, 0, 0, 0, 0, -hW_,
      1, 0, 0, 0, 0, -hL_, 0, -1, 0, 0, 0, -hL_, 0, 1, 0, -hW_, -hL_,
      -(hL_ + hW_) * mu_, mu_, mu_, -1, -hW_, hL_, -(hL_ + hW_) * mu_, mu_,
      -mu_, -1, hW_, -hL_, -(hL_ + hW_) * mu_, -mu_, mu_, -1, hW_, hL_,
      -(hL_ + hW_) * mu_, -mu_, -mu_, -1, hW_, hL_, -(hL_ + hW_) * mu_, mu_,
      mu_, 1, hW_, -hL_, -(hL_ + hW_) * mu_, mu_, -mu_, 1, -hW_, hL_,
      -(hL_ + hW_) * mu_, -mu_, mu_, 1, -hW_, -hL_, -(hL_ + hW_) * mu_, -mu_,
      -mu_, 1;

  d.Ju_.template block<17, 6>(0, k_ * 6) = d.Jtemp_;
}

template <typename Scalar>
CentroidalWrenchConeDataTpl<Scalar>::CentroidalWrenchConeDataTpl(
    const CentroidalWrenchConeResidualTpl<Scalar> *model)
    : Base(*model) {
  Jtemp_.setZero();
}

} // namespace aligator
