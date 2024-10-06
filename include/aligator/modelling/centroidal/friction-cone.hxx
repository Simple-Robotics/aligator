#pragma once

#include "aligator/modelling/centroidal/friction-cone.hpp"

namespace aligator {

template <typename Scalar>
void FrictionConeResidualTpl<Scalar>::evaluate(const ConstVectorRef &,
                                               const ConstVectorRef &u,
                                               BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.value_[0] = -u[k_ * 3 + 2] + epsilon_;
  d.value_[1] = -mu2_ * std::pow(u[k_ * 3 + 2], 2) + std::pow(u[k_ * 3], 2) +
                std::pow(u[k_ * 3 + 1], 2);
}

template <typename Scalar>
void FrictionConeResidualTpl<Scalar>::computeJacobians(const ConstVectorRef &,
                                                       const ConstVectorRef &u,
                                                       BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.Jtemp_ << 0, 0, -1, 2 * u[k_ * 3], 2 * u[k_ * 3 + 1],
      -2 * mu2_ * u[k_ * 3 + 2];
  d.Ju_.template block<2, 3>(0, k_ * 3) = d.Jtemp_;
}

template <typename Scalar>
FrictionConeDataTpl<Scalar>::FrictionConeDataTpl(
    const FrictionConeResidualTpl<Scalar> *model)
    : Base(*model) {
  Jtemp_.setZero();
}

} // namespace aligator
