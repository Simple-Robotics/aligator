/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/ode-abstract.hpp"

#include <proxsuite-nlp/modelling/spaces/multibody.hpp>
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
  using RigidConstraintModelVector = PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(
      pinocchio::RigidConstraintModelTpl<Scalar>);
  using RigidConstraintDataVector =
      PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(pinocchio::RigidConstraintData);
#pragma GCC diagnostic pop
  using ProxSettings = pinocchio::ProximalSettingsTpl<Scalar>;
  using Manifold = proxsuite::nlp::MultibodyPhaseSpace<Scalar>;

  Manifold space_;
  MatrixXs actuation_matrix_;
  RigidConstraintModelVector constraint_models_;
  ProxSettings prox_settings_;

  const Manifold &space() const { return space_; }
  int ntau() const { return space_.getModel().nv; }

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
      PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintData);

  VectorXs tau_;
  MatrixXs dtau_dx_;
  MatrixXs dtau_du_;
  RigidConstraintDataVector constraint_datas_;
  pinocchio::ProximalSettingsTpl<Scalar> settings;
  /// shared_ptr to the underlying pinocchio::DataTpl object.
  PinDataType pin_data_;
  MultibodyConstraintFwdDataTpl(
      const MultibodyConstraintFwdDynamicsTpl<Scalar> &cont_dyn);
};

} // namespace dynamics
} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/modelling/dynamics/multibody-constraint-fwd.txx"
#endif
