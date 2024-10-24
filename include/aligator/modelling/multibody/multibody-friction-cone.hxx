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
  d.value_[0] = -d.pin_data_.lambda_c[contact_id_ * 3 + 2];
  d.value_[1] = -mu_ * d.pin_data_.lambda_c[contact_id_ * 3 + 2] +
                sqrt(pow(d.pin_data_.lambda_c[contact_id_ * 3], 2) +
                     pow(d.pin_data_.lambda_c[contact_id_ * 3 + 1], 2));
}

template <typename Scalar>
void MultibodyFrictionConeResidualTpl<Scalar>::computeJacobians(
    const ConstVectorRef &, const ConstVectorRef &, BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  pinocchio::computeConstraintDynamicsDerivatives(
      pin_model_, d.pin_data_, constraint_models_, d.constraint_datas_,
      d.settings);

  d.temp_.noalias() =
      d.pin_data_.dlambda_dtau.block(contact_id_ * 3, 0, 3, pin_model_.nv) *
      actuation_matrix_;
  d.dcone_df_ << d.pin_data_.lambda_c[contact_id_ * 3] /
                     sqrt(pow(d.pin_data_.lambda_c[contact_id_ * 3], 2) +
                          pow(d.pin_data_.lambda_c[contact_id_ * 3 + 1], 2)),
      d.pin_data_.lambda_c[contact_id_ * 3 + 1] /
          sqrt(pow(d.pin_data_.lambda_c[contact_id_ * 3], 2) +
               pow(d.pin_data_.lambda_c[contact_id_ * 3 + 1], 2)),
      -mu_;

  d.Jx_.block(0, 0, 1, pin_model_.nv).noalias() =
      -d.pin_data_.dlambda_dq.block(contact_id_ * 3 + 2, 0, 1, pin_model_.nv);
  d.Jx_.block(0, pin_model_.nv, 1, pin_model_.nv).noalias() =
      -d.pin_data_.dlambda_dv.block(contact_id_ * 3 + 2, 0, 1, pin_model_.nv);
  d.Ju_.block(0, 0, 1, actuation_matrix_.cols()).noalias() =
      -d.temp_.block(2, 0, 1, actuation_matrix_.cols());

  d.Jx_.block(1, 0, 1, pin_model_.nv).noalias() =
      d.dcone_df_ *
      d.pin_data_.dlambda_dq.block(contact_id_ * 3, 0, 3, pin_model_.nv);
  d.Jx_.block(1, pin_model_.nv, 1, pin_model_.nv).noalias() =
      d.dcone_df_ *
      d.pin_data_.dlambda_dv.block(contact_id_ * 3, 0, 3, pin_model_.nv);
  d.Ju_.block(1, 0, 1, actuation_matrix_.cols()).noalias() =
      d.dcone_df_ * d.temp_;
}

template <typename Scalar>
MultibodyFrictionConeDataTpl<Scalar>::MultibodyFrictionConeDataTpl(
    const MultibodyFrictionConeResidualTpl<Scalar> *model)
    : Base(model->ndx1, model->nu, 2), pin_data_(model->pin_model_),
      tau_(model->pin_model_.nv), temp_(3, model->nu), dcone_df_(1, 3) {
  tau_.setZero();
  temp_.setZero();
  dcone_df_.setZero();

  pinocchio::initConstraintDynamics(model->pin_model_, pin_data_,
                                    model->constraint_models_);
  for (auto cm = std::begin(model->constraint_models_);
       cm != std::end(model->constraint_models_); ++cm) {
    constraint_datas_.push_back(
        pinocchio::RigidConstraintDataTpl<Scalar, 0>(*cm));
  }
}

} // namespace aligator
