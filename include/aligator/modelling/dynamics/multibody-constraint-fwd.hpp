/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2026 INRIA
#pragma once

#include "aligator/modelling/dynamics/ode-abstract.hpp"

#include "aligator/modelling/spaces/multibody.hpp"
#include <pinocchio/multibody/data.hpp>

#include <pinocchio/algorithm/proximal.hpp>

namespace aligator {
namespace dynamics {
template <typename Scalar> struct MultibodyConstraintFwdDataTpl;

/**
 * @brief   Constraint multibody forward dynamics, using Pinocchio.
 *
 */
template <typename _Scalar>
struct MultibodyConstraintFwdDynamicsTpl : ODEAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ODEAbstractTpl<Scalar>;
  using BaseData = ContinuousDynamicsDataTpl<Scalar>;
  using ContDataAbstract = ContinuousDynamicsDataTpl<Scalar>;
  using Data = MultibodyConstraintFwdDataTpl<Scalar>;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  using RigidConstraintModelVector =
      PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::RigidConstraintModelTpl<Scalar>);
  using RigidConstraintDataVector =
      PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::RigidConstraintData);
#pragma GCC diagnostic pop
  using ProxSettings = pinocchio::ProximalSettingsTpl<Scalar>;
  using Manifold = MultibodyPhaseSpace<Scalar>;

  Manifold space_;
  MatrixXs actuation_matrix_;
  RigidConstraintModelVector constraint_models_;
  ProxSettings prox_settings_;

  const Manifold &space() const { return space_; }
  int ntau() const { return space_.getModel().nv; }

  const pinocchio::ModelTpl<Scalar> &pinModel() const {
    return space_.getModel();
  }

  MultibodyConstraintFwdDynamicsTpl(
      const Manifold &state, const MatrixXs &actuation,
      const RigidConstraintModelVector &constraint_models,
      const ProxSettings &prox_settings);

  virtual void forward(const ConstVectorRef &x, const ConstVectorRef &u,
                       BaseData &data) const;
  virtual void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                        BaseData &data) const;

  shared_ptr<ContDataAbstract> createData() const;
};

template <typename Scalar>
struct MultibodyConstraintFwdDataTpl : ContinuousDynamicsDataTpl<Scalar> {
  using Base = ContinuousDynamicsDataTpl<Scalar>;
  using VectorXs = typename math_types<Scalar>::VectorXs;
  using MatrixXs = typename math_types<Scalar>::MatrixXs;
  using PinDataType = pinocchio::DataTpl<Scalar>;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  using RigidConstraintData = pinocchio::RigidConstraintDataTpl<Scalar>;
#pragma GCC diagnostic pop
  using RigidConstraintDataVector =
      PINOCCHIO_ALIGNED_STD_VECTOR(RigidConstraintData);

  VectorXs tau_;
  MatrixXs dtau_dx_;
  MatrixXs dtau_du_;
  RigidConstraintDataVector constraint_datas_;
  pinocchio::ProximalSettingsTpl<Scalar> settings;
  PinDataType pin_data_;
  explicit MultibodyConstraintFwdDataTpl(
      const MultibodyConstraintFwdDynamicsTpl<Scalar> &cont_dyn);
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct MultibodyConstraintFwdDynamicsTpl<context::Scalar>;
extern template struct MultibodyConstraintFwdDataTpl<context::Scalar>;
#endif

} // namespace dynamics
} // namespace aligator
