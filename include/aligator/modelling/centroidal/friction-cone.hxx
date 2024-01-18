#pragma once

#include "aligator/modelling/centroidal/friction-cone.hpp"

namespace aligator {

template <typename Scalar>
void FrictionConeResidualTpl<Scalar>::evaluate(const ConstVectorRef &,
                                               const ConstVectorRef &u,
                                               const ConstVectorRef &,
                                               BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.value_[0] = -u[k_ * 3 + 2];
  d.value_[1] = -mu2_ * u[k_ * 3 + 2] * u[k_ * 3 + 2] +
                (u[k_ * 3] * u[k_ * 3] + u[k_ * 3 + 1] * u[k_ * 3 + 1]);
}

template <typename Scalar>
void FrictionConeResidualTpl<Scalar>::computeJacobians(const ConstVectorRef &,
                                                       const ConstVectorRef &u,
                                                       const ConstVectorRef &,
                                                       BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.Ju_.block(0, k_ * 3, 2, 3) << 0, 0, -1, 2 * u[k_ * 3], 2 * u[k_ * 3 + 1],
      -2 * mu2_ * u[k_ * 3 + 2];
}

template <typename Scalar>
FrictionConeDataTpl<Scalar>::FrictionConeDataTpl(
    const FrictionConeResidualTpl<Scalar> *model)
    : Base(model->ndx1, model->nu, model->ndx2, 2) {}

} // namespace aligator
