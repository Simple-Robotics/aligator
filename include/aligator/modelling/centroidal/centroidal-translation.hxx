#pragma once

#include "aligator/modelling/centroidal/centroidal-translation.hpp"

namespace aligator {

template <typename Scalar>
void CentroidalCoMResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.value_ = x.head(3) - p_ref_;
}

template <typename Scalar>
void CentroidalCoMResidualTpl<Scalar>::computeJacobians(const ConstVectorRef &x,
                                                        BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.Jx_.topLeftCorner(3, 3).setIdentity();
}

template <typename Scalar>
CentroidalCoMDataTpl<Scalar>::CentroidalCoMDataTpl(
    const CentroidalCoMResidualTpl<Scalar> *model)
    : Base(model->ndx1, model->nu, model->ndx2, 3) {}

} // namespace aligator
