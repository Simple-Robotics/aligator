#pragma once

#include "aligator/modelling/centroidal/centroidal-wrapper.hpp"

namespace aligator {

template <typename Scalar>
void CentroidalWrapperResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                    BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  centroidal_cost_->evaluate(x.head(centroidal_cost_->ndx1),
                             x.tail(centroidal_cost_->nu), *d.wrapped_data_);
  d.value_ = d.wrapped_data_->value_;
}

template <typename Scalar>
void CentroidalWrapperResidualTpl<Scalar>::computeJacobians(
    const ConstVectorRef &x, BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  centroidal_cost_->computeJacobians(x.head(centroidal_cost_->ndx1),
                                     x.tail(centroidal_cost_->nu),
                                     *d.wrapped_data_);

  d.Jx_.setZero();
  d.Jx_.leftCols(centroidal_cost_->ndx1) = d.wrapped_data_->Jx_;
  d.Jx_.rightCols(centroidal_cost_->nu) = d.wrapped_data_->Ju_;

  d.Ju_.setZero();
}

template <typename Scalar>
CentroidalWrapperDataTpl<Scalar>::CentroidalWrapperDataTpl(
    const CentroidalWrapperResidualTpl<Scalar> *model)
    : Base(*model) {
  wrapped_data_ = model->centroidal_cost_->createData();
}

} // namespace aligator
