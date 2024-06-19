/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/multibody-free-fwd.hpp"

namespace aligator {
namespace dynamics {

template <typename Scalar>
MultibodyFreeFwdDynamicsTpl<Scalar>::MultibodyFreeFwdDynamicsTpl(
    const ManifoldPtr &state, const MatrixXs &actuation)
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
    const ManifoldPtr &state)
    : MultibodyFreeFwdDynamicsTpl(
          state,
          MatrixXs::Identity(state->getModel().nv, state->getModel().nv)) {}

template <typename Scalar>
void MultibodyFreeFwdDynamicsTpl<Scalar>::configure(
    CommonModelBuilderContainer &container) const {
  container.template get<MultibodyCommon>()
      .withRunAba(true)
      .withPinocchioModel(space_->getModel())
      .withActuationMatrix(actuation_matrix_);
}

template <typename Scalar>
void MultibodyFreeFwdDynamicsTpl<Scalar>::forward(const ConstVectorRef &x,
                                                  const ConstVectorRef &,
                                                  BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const pinocchio::ModelTpl<Scalar> &model = space_->getModel();
  const int nq = model.nq;
  const int nv = model.nv;
  const auto v = x.segment(nq, nv);
  d.xdot_.head(nv) = v;
  d.xdot_.segment(nv, nv) = d.multibody_data_->qdd_;
}

template <typename Scalar>
void MultibodyFreeFwdDynamicsTpl<Scalar>::dForward(const ConstVectorRef &,
                                                   const ConstVectorRef &,
                                                   BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const pinocchio::ModelTpl<Scalar> &model = space_->getModel();
  const int nv = model.nv;
  auto da_dx = d.Jx_.bottomRows(nv);
  da_dx.leftCols(nv).noalias() = d.multibody_data_->qdd_dq_;
  da_dx.rightCols(nv).noalias() = d.multibody_data_->qdd_dv_;
  d.Ju_.bottomRows(nv).noalias() = d.multibody_data_->qdd_dtau_;
}

template <typename Scalar>
shared_ptr<ContinuousDynamicsDataTpl<Scalar>>
MultibodyFreeFwdDynamicsTpl<Scalar>::createData(
    const CommonModelDataContainer &container) const {
  return allocate_shared_eigen_aligned<Data>(this, container);
}

template <typename Scalar>
MultibodyFreeFwdDataTpl<Scalar>::MultibodyFreeFwdDataTpl(
    const MultibodyFreeFwdDynamicsTpl<Scalar> *cont_dyn,
    const CommonModelDataContainerTpl<Scalar> &container)
    : Base(cont_dyn->ndx(), cont_dyn->nu()),
      multibody_data_(container.template getData<MultibodyCommon>()) {
  this->Jx_.topRightCorner(multibody_data_->nv, multibody_data_->nv)
      .setIdentity();
}

} // namespace dynamics
} // namespace aligator
