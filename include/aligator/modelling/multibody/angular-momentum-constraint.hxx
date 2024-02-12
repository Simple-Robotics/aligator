#pragma once

#include "aligator/modelling/multibody/angular-momentum-constraint.hpp"
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal-derivatives.hpp>
#include <pinocchio/algorithm/centroidal.hpp>

namespace aligator {

template <typename Scalar>
void AngularMomentumConstraintResidualTpl<Scalar>::evaluate(
    const ConstVectorRef &x, const ConstVectorRef &u, const ConstVectorRef &,
    BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const Model &model = *pin_model_;
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;

  centroidal_model_->evaluate(x.head(model.nv), u.head(centroidal_model_->nu),
                              x.head(model.nv), *d.centroidal_data_);

  pinocchio::computeCentroidalMomentum(model, pdata, x.segment(9, model.nq),
                                       x.tail(model.nv));

  d.value_ = d.centroidal_data_->value_ - pdata.hg.angular();
}

template <typename Scalar>
void AngularMomentumConstraintResidualTpl<Scalar>::computeJacobians(
    const ConstVectorRef &x, const ConstVectorRef &u, const ConstVectorRef &,
    BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const Model &model = *pin_model_;
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;

  centroidal_model_->computeJacobians(x.head(model.nv),
                                      u.head(centroidal_model_->nu),
                                      x.head(model.nv), *d.centroidal_data_);

  pinocchio::computeCentroidalDynamicsDerivatives(
      model, pdata, x.segment(9, model.nq), x.tail(model.nv), u.tail(model.nv),
      d.dh_dq_, d.dhdot_dq_, d.dhdot_dv_, d.dhdot_da_);

  d.Jx_.template leftCols<9>() = d.centroidal_data_->Jx_;
  d.Jx_.block(0, 9, 3, model.nv) = -d.dh_dq_.bottomRows(3);
  d.Jx_.rightCols(model.nv) = -d.dhdot_da_.bottomRows(3);

  d.Ju_.leftCols(centroidal_model_->nu) = d.centroidal_data_->Ju_;
}

template <typename Scalar>
AngularMomentumConstraintDataTpl<Scalar>::AngularMomentumConstraintDataTpl(
    const AngularMomentumConstraintResidualTpl<Scalar> *model)
    : Base(model->ndx1, model->nu, model->ndx2, 3),
      pin_data_(*model->pin_model_), dh_dq_(6, model->pin_model_->nv),
      dhdot_dq_(6, model->pin_model_->nv), dhdot_dv_(6, model->pin_model_->nv),
      dhdot_da_(6, model->pin_model_->nv) {
  centroidal_data_ = model->centroidal_model_->createData();
  dh_dq_.setZero();
  dhdot_dq_.setZero();
  dhdot_dv_.setZero();
  dhdot_da_.setZero();
}

} // namespace aligator
