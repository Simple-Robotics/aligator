#pragma once

#include "aligator/modelling/multibody/centroidal-momentum.hpp"

#include <pinocchio/algorithm/centroidal-derivatives.hpp>
#include <pinocchio/algorithm/centroidal.hpp>

namespace aligator {

template <typename Scalar>
void CentroidalMomentumResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                     BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;

  const auto q = x.head(pin_model_.nq);
  const auto v = x.tail(pin_model_.nv);

  pinocchio::ccrba(pin_model_, pdata, q, v); // Compute Ag

  d.value_ = pdata.Ag * v - h_ref_;
}

template <typename Scalar>
void CentroidalMomentumResidualTpl<Scalar>::computeJacobians(
    const ConstVectorRef &x, BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;

  const auto q = x.head(pin_model_.nq);
  const auto v = x.tail(pin_model_.nv);

  pinocchio::computeCentroidalDynamicsDerivatives(
      pin_model_, pdata, q, v, Eigen::VectorXd::Zero(pin_model_.nv), d.dh_dq_,
      d.dhdot_dq_, d.dhdot_dv_, d.dhdot_da_);

  d.Jx_.leftCols(pin_model_.nv) = d.dh_dq_;
  d.Jx_.rightCols(pin_model_.nv) = pdata.Ag;
}

template <typename Scalar>
CentroidalMomentumDataTpl<Scalar>::CentroidalMomentumDataTpl(
    const CentroidalMomentumResidualTpl<Scalar> *model)
    : Base(model->ndx1, model->nu, 6), pin_data_(model->pin_model_),
      dh_dq_(6, model->pin_model_.nv), dhdot_dq_(6, model->pin_model_.nv),
      dhdot_dv_(6, model->pin_model_.nv), dhdot_da_(6, model->pin_model_.nv) {
  dh_dq_.setZero();
  dhdot_dq_.setZero();
  dhdot_dv_.setZero();
  dhdot_da_.setZero();
}

} // namespace aligator
