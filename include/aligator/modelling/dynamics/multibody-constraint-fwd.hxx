/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/multibody-constraint-fwd.hpp"

#include <pinocchio/algorithm/constrained-dynamics.hpp>
#include <pinocchio/algorithm/constrained-dynamics-derivatives.hpp>

namespace aligator {
namespace dynamics {
template <typename Scalar>
MultibodyConstraintFwdDynamicsTpl<Scalar>::MultibodyConstraintFwdDynamicsTpl(
    const ManifoldPtr &state, const MatrixXs &actuation,
    const RigidConstraintModelVector &constraint_models,
    const ProxSettings &prox_settings)
    : Base(state, (int)actuation.cols()), space_(state),
      actuation_matrix_(actuation), constraint_models_(constraint_models),
      prox_settings_(prox_settings) {
  const int nv = state->getModel().nv;
  if (nv != actuation.rows()) {
    ALIGATOR_DOMAIN_ERROR(
        fmt::format("actuation matrix should have number of rows = pinocchio "
                    "model nv ({} and {}).",
                    actuation.rows(), nv));
  }
}

template <typename Scalar>
void MultibodyConstraintFwdDynamicsTpl<Scalar>::configure(
    CommonModelBuilderContainer &container) const {
  container.template get<MultibodyConstraintCommon>()
      .withRunAba(true)
      .withPinocchioModel(space_->getModel())
      .withActuationMatrix(actuation_matrix_)
      .withProxSettings(prox_settings_)
      .withConstraintModels(constraint_models_);
}

template <typename Scalar>
void MultibodyConstraintFwdDynamicsTpl<Scalar>::forward(const ConstVectorRef &x,
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
void MultibodyConstraintFwdDynamicsTpl<Scalar>::dForward(const ConstVectorRef &,
                                                         const ConstVectorRef &,
                                                         BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const pinocchio::ModelTpl<Scalar> &model = space_->getModel();
  const int nv = model.nv;
  d.Jx_.bottomRows(nv).leftCols(nv) = d.multibody_data_->pin_data_.ddq_dq;
  d.Jx_.bottomRows(nv).rightCols(nv) = d.multibody_data_->pin_data_.ddq_dv;
  d.Ju_.bottomRows(nv) = d.multibody_data_->qdd_dtau_;
}

template <typename Scalar>
shared_ptr<ContinuousDynamicsDataTpl<Scalar>>
MultibodyConstraintFwdDynamicsTpl<Scalar>::createData(
    const CommonModelDataContainer &container) const {
  return allocate_shared_eigen_aligned<Data>(*this, container);
}

template <typename Scalar>
MultibodyConstraintFwdDataTpl<Scalar>::MultibodyConstraintFwdDataTpl(
    const MultibodyConstraintFwdDynamicsTpl<Scalar> &cont_dyn,
    const CommonModelDataContainer &container)
    : Base(cont_dyn.ndx(), cont_dyn.nu()),
      multibody_data_(container.template getData<MultibodyConstraintCommon>()) {
  this->Jx_.topRightCorner(multibody_data_->nv, multibody_data_->nv)
      .setIdentity();
}

} // namespace dynamics
} // namespace aligator
