#pragma once

#include "aligator/modelling/multibody/kinodynamics-wrapper.hpp"

namespace aligator {

template <typename Scalar>
void KinodynamicsWrapperResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                      BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  multibody_cost_->evaluate(x.tail(nx_), *d.wrapped_data_);
  d.value_ = d.wrapped_data_->value_;
}

template <typename Scalar>
void KinodynamicsWrapperResidualTpl<Scalar>::computeJacobians(
    const ConstVectorRef &x, BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  multibody_cost_->computeJacobians(x.tail(nx_), *d.wrapped_data_);

  d.Jx_.rightCols(multibody_cost_->ndx1) = d.wrapped_data_->Jx_;
}

template <typename Scalar>
KinodynamicsWrapperDataTpl<Scalar>::KinodynamicsWrapperDataTpl(
    const KinodynamicsWrapperResidualTpl<Scalar> *model)
    : Base(model->ndx1, model->nu, model->ndx2, model->nr) {
  wrapped_data_ = model->multibody_cost_->createData();
  this->Jx_.setZero();
  this->Ju_.setZero();
}

} // namespace aligator
