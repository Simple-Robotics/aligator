/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/multibody-constraint-fwd.hpp"

#include <pinocchio/algorithm/constrained-dynamics.hpp>
#include <pinocchio/algorithm/constrained-dynamics-derivatives.hpp>

namespace aligator {
namespace dynamics {
template <typename Scalar>
MultibodyConstraintFwdDynamicsTpl<Scalar>::MultibodyConstraintFwdDynamicsTpl(
    const Manifold &state, const MatrixXs &actuation,
    const RigidConstraintModelVector &constraint_models,
    const ProxSettings &prox_settings)
    : Base(state, (int)actuation.cols()), space_(state),
      actuation_matrix_(actuation), constraint_models_(constraint_models),
      prox_settings_(prox_settings) {
  const int nv = state.getModel().nv;
  if (nv != actuation.rows()) {
    ALIGATOR_DOMAIN_ERROR(
        fmt::format("actuation matrix should have number of rows = pinocchio "
                    "model nv ({} and {}).",
                    actuation.rows(), nv));
  }
}

template <typename Scalar>
void MultibodyConstraintFwdDynamicsTpl<Scalar>::forward(const ConstVectorRef &x,
                                                        const ConstVectorRef &u,
                                                        BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  d.tau_ = actuation_matrix_ * u;
  const pinocchio::ModelTpl<Scalar> &model = space_.getModel();
  const int nq = model.nq;
  const int nv = model.nv;
  const ConstVectorRef q = x.head(nq);
  const ConstVectorRef v = x.segment(nq, nv);
  const ConstVectorRef tau = d.tau_;
  d.xdot_.head(nv) = v;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  d.xdot_.segment(nv, nv) = pinocchio::constraintDynamics(
      model, d.pin_data_, q, v, tau, constraint_models_, d.constraint_datas_,
      d.settings);
#pragma GCC diagnostic pop
}

template <typename Scalar>
void MultibodyConstraintFwdDynamicsTpl<Scalar>::dForward(const ConstVectorRef &,
                                                         const ConstVectorRef &,
                                                         BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const pinocchio::ModelTpl<Scalar> &model = space_.getModel();
  const int nv = model.nv;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  pinocchio::computeConstraintDynamicsDerivatives(
      model, d.pin_data_, constraint_models_, d.constraint_datas_, d.settings);
#pragma GCC diagnostic pop
  d.Jx_.bottomRows(nv).leftCols(nv) = d.pin_data_.ddq_dq;
  d.Jx_.bottomRows(nv).rightCols(nv) = d.pin_data_.ddq_dv;
  d.Ju_.bottomRows(nv) = d.pin_data_.ddq_dtau * d.dtau_du_;
}

template <typename Scalar>
shared_ptr<ContinuousDynamicsDataTpl<Scalar>>
MultibodyConstraintFwdDynamicsTpl<Scalar>::createData() const {
  return std::make_shared<Data>(*this);
}

template <typename Scalar>
MultibodyConstraintFwdDataTpl<Scalar>::MultibodyConstraintFwdDataTpl(
    const MultibodyConstraintFwdDynamicsTpl<Scalar> &cont_dyn)
    : Base(cont_dyn.ndx(), cont_dyn.nu()), tau_(cont_dyn.space_.getModel().nv),
      dtau_dx_(cont_dyn.ntau(), cont_dyn.ndx()),
      dtau_du_(cont_dyn.actuation_matrix_), settings(cont_dyn.prox_settings_),
      pin_data_() {
  tau_.setZero();

  const pinocchio::ModelTpl<Scalar> &model = cont_dyn.space_.getModel();
  pin_data_ = PinDataType(model);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  pinocchio::initConstraintDynamics(model, pin_data_,
                                    cont_dyn.constraint_models_);
#pragma GCC diagnostic pop
  for (auto cm = std::begin(cont_dyn.constraint_models_);
       cm != std::end(cont_dyn.constraint_models_); ++cm) {
    constraint_datas_.emplace_back(*cm);
  }
  this->Jx_.topRightCorner(model.nv, model.nv).setIdentity();
}
} // namespace dynamics
} // namespace aligator
