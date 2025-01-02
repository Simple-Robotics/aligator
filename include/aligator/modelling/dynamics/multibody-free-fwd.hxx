/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/multibody-free-fwd.hpp"

#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/aba-derivatives.hpp>
#include <pinocchio/multibody/fwd.hpp>

namespace aligator {
namespace dynamics {

template <typename Scalar>
MultibodyFreeFwdDynamicsTpl<Scalar>::MultibodyFreeFwdDynamicsTpl(
    const Manifold &state, const MatrixXs &actuation)
    : Base(state, (int)actuation.cols()), space_(state),
      actuation_matrix_(actuation), lu_decomp_(actuation_matrix_) {
  const int nv = space().getModel().nv;
  if (nv != actuation.rows()) {
    ALIGATOR_DOMAIN_ERROR(
        fmt::format("actuation matrix should have number of rows = pinocchio "
                    "model nv ({} and {}).",
                    actuation.rows(), nv));
  }
  act_matrix_rank = lu_decomp_.rank();
}

template <typename Scalar>
MultibodyFreeFwdDynamicsTpl<Scalar>::MultibodyFreeFwdDynamicsTpl(
    const Manifold &state)
    : MultibodyFreeFwdDynamicsTpl(
          state, MatrixXs::Identity(state.getModel().nv, state.getModel().nv)) {
}

template <typename Scalar>
void MultibodyFreeFwdDynamicsTpl<Scalar>::forward(const ConstVectorRef &x,
                                                  const ConstVectorRef &u,
                                                  BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  d.tau_.noalias() = actuation_matrix_ * u;
  const pinocchio::ModelTpl<Scalar> &model = space_.getModel();
  const int nq = model.nq;
  const int nv = model.nv;
  const ConstVectorRef q = x.head(nq);
  const ConstVectorRef v = x.segment(nq, nv);
  const ConstVectorRef tau = d.tau_;
  d.xdot_.head(nv) = v;
  d.xdot_.segment(nv, nv) = pinocchio::aba(model, d.pin_data_, q, v, tau,
                                           pinocchio::Convention::LOCAL);
}

template <typename Scalar>
void MultibodyFreeFwdDynamicsTpl<Scalar>::dForward(const ConstVectorRef &x,
                                                   const ConstVectorRef &,
                                                   BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const pinocchio::ModelTpl<Scalar> &model = space_.getModel();
  const int nq = model.nq;
  const int nv = model.nv;
  auto da_dx = d.Jx_.bottomRows(nv);
  const ConstVectorRef q = x.head(nq);
  const ConstVectorRef v = x.tail(nv);
  const ConstVectorRef tau = d.tau_;
  pinocchio::computeABADerivatives(model, d.pin_data_, q, v, tau,
                                   pinocchio::make_ref(da_dx.leftCols(nv)),
                                   pinocchio::make_ref(da_dx.rightCols(nv)),
                                   pinocchio::make_ref(d.pin_data_.Minv));
  d.Ju_.bottomRows(nv) = d.pin_data_.Minv * d.dtau_du_;
}

template <typename Scalar>
shared_ptr<ContinuousDynamicsDataTpl<Scalar>>
MultibodyFreeFwdDynamicsTpl<Scalar>::createData() const {
  return std::make_shared<Data>(this);
}

template <typename Scalar>
MultibodyFreeFwdDataTpl<Scalar>::MultibodyFreeFwdDataTpl(
    const MultibodyFreeFwdDynamicsTpl<Scalar> *cont_dyn)
    : Base(cont_dyn->ndx(), cont_dyn->nu()),
      tau_(cont_dyn->space_.getModel().nv),
      dtau_dx_(cont_dyn->ntau(), cont_dyn->ndx()),
      dtau_du_(cont_dyn->actuation_matrix_), pin_data_() {
  tau_.setZero();
  const pinocchio::ModelTpl<Scalar> &model = cont_dyn->space_.getModel();
  pin_data_ = PinDataType(model);
  this->Jx_.topRightCorner(model.nv, model.nv).setIdentity();
}
} // namespace dynamics
} // namespace aligator
