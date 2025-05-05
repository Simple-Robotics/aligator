#pragma once

#include "aligator/modelling/multibody/contact-force.hpp"

#include <pinocchio/algorithm/constrained-dynamics.hpp>
#include <pinocchio/algorithm/constrained-dynamics-derivatives.hpp>

namespace aligator {

template <typename Scalar>
void ContactForceResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                               const ConstVectorRef &u,
                                               BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  const ConstVectorRef q = x.head(pin_model_.nq);
  const ConstVectorRef v = x.tail(pin_model_.nv);

  d.tau_.noalias() = actuation_matrix_ * u;
  pinocchio::constraintDynamics(
      pin_model_, d.pin_data_, q, v, pinocchio::make_const_ref(d.tau_),
      constraint_models_, d.constraint_datas_, d.settings);
  d.value_ =
      d.pin_data_.lambda_c.segment(contact_id_ * force_size_, force_size_) -
      fref_;
}

template <typename Scalar>
void ContactForceResidualTpl<Scalar>::computeJacobians(const ConstVectorRef &,
                                                       const ConstVectorRef &,
                                                       BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  pinocchio::computeConstraintDynamicsDerivatives(
      pin_model_, d.pin_data_, constraint_models_, d.constraint_datas_,
      d.settings);
  d.Jx_.leftCols(pin_model_.nv) = d.pin_data_.dlambda_dq.block(
      contact_id_ * force_size_, 0, force_size_, pin_model_.nv);
  d.Jx_.rightCols(pin_model_.nv) = d.pin_data_.dlambda_dv.block(
      contact_id_ * force_size_, 0, force_size_, pin_model_.nv);
  d.Ju_.noalias() = d.pin_data_.dlambda_dtau.block(contact_id_ * force_size_, 0,
                                                   force_size_, pin_model_.nv) *
                    actuation_matrix_;
}

template <typename Scalar>
ContactForceDataTpl<Scalar>::ContactForceDataTpl(
    const ContactForceResidualTpl<Scalar> *model)
    : Base(model->ndx1, model->nu, (int)model->force_size_),
      pin_data_(model->pin_model_), tau_(model->pin_model_.nv) {
  tau_.setZero();

  pinocchio::initConstraintDynamics(model->pin_model_, pin_data_,
                                    model->constraint_models_);
  for (auto cm = std::begin(model->constraint_models_);
       cm != std::end(model->constraint_models_); ++cm) {
    constraint_datas_.push_back(
        pinocchio::RigidConstraintDataTpl<Scalar, 0>(*cm));
  }
}

} // namespace aligator
