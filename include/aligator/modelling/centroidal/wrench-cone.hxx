#pragma once

#include "aligator/modelling/centroidal/wrench-cone.hpp"

namespace aligator {

template <typename Scalar>
void WrenchConeResidualTpl<Scalar>::evaluate(const ConstVectorRef &,
                                             const ConstVectorRef &u,
                                             const ConstVectorRef &,
                                             BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.value_[0] = -u[k_ * 6] - mu_ * u[k_ * 6 + 2];
  d.value_[1] = +u[k_ * 6] - mu_ * u[k_ * 6 + 2];
  d.value_[2] = -u[k_ * 6 + 1] - mu_ * u[k_ * 6 + 2];
  d.value_[3] = +u[k_ * 6 + 1] - mu_ * u[k_ * 6 + 2];
  d.value_[4] = -W_ * u[k_ * 6 + 2] - u[k_ * 6 + 3];
  d.value_[5] = -W_ * u[k_ * 6 + 2] + u[k_ * 6 + 3];
  d.value_[6] = -L_ * u[k_ * 6 + 2] - u[k_ * 6 + 4];
  d.value_[7] = -L_ * u[k_ * 6 + 2] + u[k_ * 6 + 4];

  d.value_[8] = -W_ * u[k_ * 6] - L_ * u[k_ * 6 + 1] -
                (W_ + L_) * mu_ * u[k_ * 6 + 2] + mu_ * u[k_ * 6 + 3] +
                mu_ * u[k_ * 6 + 4] - u[k_ * 6 + 5];
  d.value_[9] = -W_ * u[k_ * 6] + L_ * u[k_ * 6 + 1] -
                (W_ + L_) * mu_ * u[k_ * 6 + 2] + mu_ * u[k_ * 6 + 3] -
                mu_ * u[k_ * 6 + 4] - u[k_ * 6 + 5];
  d.value_[10] = W_ * u[k_ * 6] - L_ * u[k_ * 6 + 1] -
                 (W_ + L_) * mu_ * u[k_ * 6 + 2] - mu_ * u[k_ * 6 + 3] +
                 mu_ * u[k_ * 6 + 4] - u[k_ * 6 + 5];
  d.value_[11] = W_ * u[k_ * 6] + L_ * u[k_ * 6 + 1] -
                 (W_ + L_) * mu_ * u[k_ * 6 + 2] - mu_ * u[k_ * 6 + 3] -
                 mu_ * u[k_ * 6 + 4] - u[k_ * 6 + 5];

  d.value_[12] = W_ * u[k_ * 6] + L_ * u[k_ * 6 + 1] -
                 (W_ + L_) * mu_ * u[k_ * 6 + 2] + mu_ * u[k_ * 6 + 3] +
                 mu_ * u[k_ * 6 + 4] + u[k_ * 6 + 5];
  d.value_[13] = W_ * u[k_ * 6] - L_ * u[k_ * 6 + 1] -
                 (W_ + L_) * mu_ * u[k_ * 6 + 2] + mu_ * u[k_ * 6 + 3] -
                 mu_ * u[k_ * 6 + 4] + u[k_ * 6 + 5];
  d.value_[14] = -W_ * u[k_ * 6] + L_ * u[k_ * 6 + 1] -
                 (W_ + L_) * mu_ * u[k_ * 6 + 2] - mu_ * u[k_ * 6 + 3] +
                 mu_ * u[k_ * 6 + 4] + u[k_ * 6 + 5];
  d.value_[15] = -W_ * u[k_ * 6] - L_ * u[k_ * 6 + 1] -
                 (W_ + L_) * mu_ * u[k_ * 6 + 2] - mu_ * u[k_ * 6 + 3] -
                 mu_ * u[k_ * 6 + 4] + u[k_ * 6 + 5];
}

template <typename Scalar>
void WrenchConeResidualTpl<Scalar>::computeJacobians(const ConstVectorRef &,
                                                     const ConstVectorRef &,
                                                     const ConstVectorRef &,
                                                     BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.Jtemp_ << -1, 0, -mu_, 0, 0, 0, 1, 0, -mu_, 0, 0, 0, 0, -1, -mu_, 0, 0, 0,
      0, 1, -mu_, 0, 0, 0, 0, 0, -W_, -1, 0, 0, 0, 0, -W_, 1, 0, 0, 0, 0, -L_,
      0, -1, 0, 0, 0, -L_, 0, 1, 0, -W_, -L_, -(L_ + W_) * mu_, mu_, mu_, -1,
      -W_, L_, -(L_ + W_) * mu_, mu_, -mu_, -1, W_, -L_, -(L_ + W_) * mu_, -mu_,
      mu_, -1, W_, L_, -(L_ + W_) * mu_, -mu_, -mu_, -1, W_, L_,
      -(L_ + W_) * mu_, mu_, mu_, 1, W_, -L_, -(L_ + W_) * mu_, mu_, -mu_, 1,
      -W_, L_, -(L_ + W_) * mu_, -mu_, mu_, 1, -W_, -L_, -(L_ + W_) * mu_, -mu_,
      -mu_, 1;

  d.Ju_.template block<16, 6>(0, k_ * 6) = d.Jtemp_;
}

template <typename Scalar>
WrenchConeDataTpl<Scalar>::WrenchConeDataTpl(
    const WrenchConeResidualTpl<Scalar> *model)
    : Base(model->ndx1, model->nu, model->ndx2, 16) {
  Jtemp_.setZero();
}

} // namespace aligator
