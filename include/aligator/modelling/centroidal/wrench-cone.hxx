#pragma once

#include "aligator/modelling/centroidal/wrench-cone.hpp"

namespace aligator {

template <typename Scalar>
void WrenchConeResidualTpl<Scalar>::evaluate(const ConstVectorRef &,
                                             const ConstVectorRef &u,
                                             const ConstVectorRef &,
                                             BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.value_[0] = -u[k_ * 6 + 2] + epsilon_;
  d.value_[1] = -mu2_ * std::pow(u[k_ * 6 + 2], 2) + std::pow(u[k_ * 6], 2) +
                std::pow(u[k_ * 6 + 1], 2);
  d.value_[2] = -W_ * u[k_ * 6 + 2] + u[k_ * 6 + 3] + epsilon_;
  d.value_[3] = -W_ * u[k_ * 6 + 2] - u[k_ * 6 + 3] + epsilon_;
  d.value_[4] = -L_ * u[k_ * 6 + 2] + u[k_ * 6 + 4] + epsilon_;
  d.value_[5] = -L_ * u[k_ * 6 + 2] - u[k_ * 6 + 4] + epsilon_;
}

template <typename Scalar>
void WrenchConeResidualTpl<Scalar>::computeJacobians(const ConstVectorRef &,
                                                     const ConstVectorRef &u,
                                                     const ConstVectorRef &,
                                                     BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.Jtemp_ << 0, 0, -1, 0, 0, 0, 2 * u[k_ * 6], 2 * u[k_ * 6 + 1],
      -2 * mu2_ * u[k_ * 6 + 2], 0, 0, 0, 0, 0, -W_, 1, 0, 0, 0, 0, -W_, -1, 0,
      0, 0, 0, -L_, 0, 1, 0, 0, 0, -L_, 0, -1, 0;
  d.Ju_.template block<6, 6>(0, k_ * 6) = d.Jtemp_;
}

template <typename Scalar>
WrenchConeDataTpl<Scalar>::WrenchConeDataTpl(
    const WrenchConeResidualTpl<Scalar> *model)
    : Base(model->ndx1, model->nu, model->ndx2, 6) {
  Jtemp_.setZero();
}

} // namespace aligator
