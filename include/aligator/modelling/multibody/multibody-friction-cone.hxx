#pragma once

#include "aligator/modelling/multibody/multibody-friction-cone.hpp"

#include <pinocchio/algorithm/constrained-dynamics.hpp>
#include <pinocchio/algorithm/constrained-dynamics-derivatives.hpp>

namespace aligator {

template <typename Scalar>
void MultibodyFrictionConeResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                        const ConstVectorRef &u,
                                                        BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  const auto q = x.head(pin_model_.nq);
  const auto v = x.tail(pin_model_.nv);

  d.tau_.noalias() = actuation_matrix_ * u;
  pinocchio::constraintDynamics(pin_model_, d.pin_data_, q, v, d.tau_,
                                constraint_models_, d.constraint_datas_,
                                d.settings);

  // Unilateral contact
  d.value_.noalias() =
      Acone_ * d.pin_data_.lambda_c.segment(contact_id_ * 3, 3);
}

template <typename Scalar>
void MultibodyFrictionConeResidualTpl<Scalar>::computeJacobians(
    const ConstVectorRef &, const ConstVectorRef &, BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  pinocchio::computeConstraintDynamicsDerivatives(
      pin_model_, d.pin_data_, constraint_models_, d.constraint_datas_,
      d.settings);

  d.Jx_.leftCols(pin_model_.nv).noalias() =
      Acone_ *
      d.pin_data_.dlambda_dq.block(contact_id_ * 3, 0, 3, pin_model_.nv);
  d.Jx_.rightCols(pin_model_.nv).noalias() =
      Acone_ *
      d.pin_data_.dlambda_dv.block(contact_id_ * 3, 0, 3, pin_model_.nv);
  d.temp_.noalias() =
      d.pin_data_.dlambda_dtau.block(contact_id_ * 3, 0, 3, pin_model_.nv) *
      actuation_matrix_;
  d.Ju_.noalias() = Acone_ * d.temp_;
}

template <typename Scalar>
MultibodyFrictionConeDataTpl<Scalar>::MultibodyFrictionConeDataTpl(
    const MultibodyFrictionConeResidualTpl<Scalar> *model)
    : Base(model->ndx1, model->nu, 5), pin_data_(model->pin_model_),
      tau_(model->pin_model_.nv), temp_(3, model->nu) {
  tau_.setZero();
  temp_.setZero();

  pinocchio::initConstraintDynamics(model->pin_model_, pin_data_,
                                    model->constraint_models_);
  for (auto cm = std::begin(model->constraint_models_);
       cm != std::end(model->constraint_models_); ++cm) {
    constraint_datas_.push_back(
        pinocchio::RigidConstraintDataTpl<Scalar, 0>(*cm));
  }
}

} // namespace aligator
