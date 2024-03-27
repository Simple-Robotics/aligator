#pragma once

#include "aligator/modelling/centroidal/wrench-cone.hpp"

namespace aligator {

template <typename Scalar>
void WrenchConeResidualTpl<Scalar>::evaluate(const ConstVectorRef &,
                                             const ConstVectorRef &u,
                                             const ConstVectorRef &,
                                             BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  // Coulomb friction inequalities
  d.value_[0] = -u[k_ * 6] - mu_ * u[k_ * 6 + 2];
  d.value_[1] = +u[k_ * 6] - mu_ * u[k_ * 6 + 2];
  d.value_[2] = -u[k_ * 6 + 1] - mu_ * u[k_ * 6 + 2];
  d.value_[3] = +u[k_ * 6 + 1] - mu_ * u[k_ * 6 + 2];

  // Local CoP inequalities
  d.value_[4] = -hW_ * u[k_ * 6 + 2] - u[k_ * 6 + 3];
  d.value_[5] = -hW_ * u[k_ * 6 + 2] + u[k_ * 6 + 3];
  d.value_[6] = -hL_ * u[k_ * 6 + 2] - u[k_ * 6 + 4];
  d.value_[7] = -hL_ * u[k_ * 6 + 2] + u[k_ * 6 + 4];

  // z-torque limits
  d.value_[8] = -hW_ * u[k_ * 6] - hL_ * u[k_ * 6 + 1] -
                (hW_ + hL_) * mu_ * u[k_ * 6 + 2] + mu_ * u[k_ * 6 + 3] +
                mu_ * u[k_ * 6 + 4] - u[k_ * 6 + 5];
  d.value_[9] = -hW_ * u[k_ * 6] + hL_ * u[k_ * 6 + 1] -
                (hW_ + hL_) * mu_ * u[k_ * 6 + 2] + mu_ * u[k_ * 6 + 3] -
                mu_ * u[k_ * 6 + 4] - u[k_ * 6 + 5];
  d.value_[10] = hW_ * u[k_ * 6] - hL_ * u[k_ * 6 + 1] -
                 (hW_ + hL_) * mu_ * u[k_ * 6 + 2] - mu_ * u[k_ * 6 + 3] +
                 mu_ * u[k_ * 6 + 4] - u[k_ * 6 + 5];
  d.value_[11] = hW_ * u[k_ * 6] + hL_ * u[k_ * 6 + 1] -
                 (hW_ + hL_) * mu_ * u[k_ * 6 + 2] - mu_ * u[k_ * 6 + 3] -
                 mu_ * u[k_ * 6 + 4] - u[k_ * 6 + 5];

  d.value_[12] = hW_ * u[k_ * 6] + hL_ * u[k_ * 6 + 1] -
                 (hW_ + hL_) * mu_ * u[k_ * 6 + 2] + mu_ * u[k_ * 6 + 3] +
                 mu_ * u[k_ * 6 + 4] + u[k_ * 6 + 5];
  d.value_[13] = hW_ * u[k_ * 6] - hL_ * u[k_ * 6 + 1] -
                 (hW_ + hL_) * mu_ * u[k_ * 6 + 2] + mu_ * u[k_ * 6 + 3] -
                 mu_ * u[k_ * 6 + 4] + u[k_ * 6 + 5];
  d.value_[14] = -hW_ * u[k_ * 6] + hL_ * u[k_ * 6 + 1] -
                 (hW_ + hL_) * mu_ * u[k_ * 6 + 2] - mu_ * u[k_ * 6 + 3] +
                 mu_ * u[k_ * 6 + 4] + u[k_ * 6 + 5];
  d.value_[15] = -hW_ * u[k_ * 6] - hL_ * u[k_ * 6 + 1] -
                 (hW_ + hL_) * mu_ * u[k_ * 6 + 2] - mu_ * u[k_ * 6 + 3] -
                 mu_ * u[k_ * 6 + 4] + u[k_ * 6 + 5];
}

template <typename Scalar>
void WrenchConeResidualTpl<Scalar>::computeJacobians(const ConstVectorRef &,
                                                     const ConstVectorRef &,
                                                     const ConstVectorRef &,
                                                     BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.Jtemp_ << -1, 0, -mu_, 0, 0, 0, 1, 0, -mu_, 0, 0, 0, 0, -1, -mu_, 0, 0, 0,
      0, 1, -mu_, 0, 0, 0, 0, 0, -hW_, -1, 0, 0, 0, 0, -hW_, 1, 0, 0, 0, 0,
      -hL_, 0, -1, 0, 0, 0, -hL_, 0, 1, 0, -hW_, -hL_, -(hL_ + hW_) * mu_, mu_,
      mu_, -1, -hW_, hL_, -(hL_ + hW_) * mu_, mu_, -mu_, -1, hW_, -hL_,
      -(hL_ + hW_) * mu_, -mu_, mu_, -1, hW_, hL_, -(hL_ + hW_) * mu_, -mu_,
      -mu_, -1, hW_, hL_, -(hL_ + hW_) * mu_, mu_, mu_, 1, hW_, -hL_,
      -(hL_ + hW_) * mu_, mu_, -mu_, 1, -hW_, hL_, -(hL_ + hW_) * mu_, -mu_,
      mu_, 1, -hW_, -hL_, -(hL_ + hW_) * mu_, -mu_, -mu_, 1;

  d.Ju_.template block<16, 6>(0, k_ * 6) = d.Jtemp_;
}

template <typename Scalar>
WrenchConeDataTpl<Scalar>::WrenchConeDataTpl(
    const WrenchConeResidualTpl<Scalar> *model)
    : Base(model->ndx1, model->nu, model->ndx2, 16) {
  Jtemp_.setZero();
}

} // namespace aligator
