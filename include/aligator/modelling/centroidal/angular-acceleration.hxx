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
  auto it = contact_map_.begin();
  for (std::size_t i = 0; i < nk_; i++) {
    if (it->first) {
      d.value_[0] += (it->second[1] - x[1]) * u[i * 3 + 2] -
                     (it->second[2] - x[2]) * u[i * 3 + 1];
      d.value_[1] += (it->second[2] - x[2]) * u[i * 3] -
                     (it->second[0] - x[0]) * u[i * 3 + 2];
      d.value_[2] += (it->second[0] - x[0]) * u[i * 3 + 1] -
                     (it->second[1] - x[1]) * u[i * 3];
    }
    it++;
  }
}

template <typename Scalar>
void AngularAccelerationResidualTpl<Scalar>::computeJacobians(
    const ConstVectorRef &x, const ConstVectorRef &u, const ConstVectorRef &,
    BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.Jx_.setZero();
  d.Ju_.setZero();
  auto it = contact_map_.begin();
  for (std::size_t i = 0; i < nk_; i++) {
    if (it->first) {
      d.Jx_(0, 1) -= u[i * 3 + 2];
      d.Jx_(0, 2) += u[i * 3 + 1];
      d.Jx_(1, 0) += u[i * 3 + 2];
      d.Jx_(1, 2) -= u[i * 3];
      d.Jx_(2, 0) -= u[i * 3 + 1];
      d.Jx_(2, 1) += u[i * 3];

      d.Ju_.block(0, 3 * i, 3, 3) << 0.0, -(it->second[2] - x[2]),
          (it->second[1] - x[1]), (it->second[2] - x[2]), 0.0,
          -(it->second[0] - x[0]), -(it->second[1] - x[1]),
          (it->second[0] - x[0]), 0.0;
    }
    it++;
  }
}

template <typename Scalar>
AngularAccelerationDataTpl<Scalar>::AngularAccelerationDataTpl(
    const AngularAccelerationResidualTpl<Scalar> *model)
    : Base(model->ndx1, model->nu, model->ndx2, 3) {}

} // namespace aligator
