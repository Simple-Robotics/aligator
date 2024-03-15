/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/ode-abstract.hpp"
#include "aligator/modelling/dynamics/multibody-constraint-common.hpp"

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
  ALIGATOR_ODE_TYPEDEFS(Scalar, MultibodyConstraintFwdDataTpl);

  using RigidConstraintModelVector = PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(
      pinocchio::RigidConstraintModel);
  using RigidConstraintDataVector =
      PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(pinocchio::RigidConstraintData);
  using ProxSettings = pinocchio::ProximalSettingsTpl<Scalar>;
  using Manifold = proxsuite::nlp::MultibodyPhaseSpace<Scalar>;
  using MultibodyConstraintCommon = MultibodyConstraintCommonTpl<Scalar>;

  using ManifoldPtr = shared_ptr<Manifold>;
  ManifoldPtr space_;
  MatrixXs actuation_matrix_;
  RigidConstraintModelVector constraint_models_;
  ProxSettings prox_settings_;

  const Manifold &space() const { return *space_; }
  int ntau() const { return space_->getModel().nv; }

  MultibodyConstraintFwdDynamicsTpl(
      const ManifoldPtr &state, const MatrixXs &actuation,
      const RigidConstraintModelVector &constraint_models,
      const ProxSettings &prox_settings);

  void configure(CommonModelBuilderContainer &container) const override;
  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               BaseData &data) const override;
  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) const override;

  shared_ptr<ContinuousData>
  createData(const CommonModelDataContainer &container) const override;
  shared_ptr<ContinuousData> createData() const override {
    ALIGATOR_RUNTIME_ERROR("createData can't be called without arguments");
  }
};

template <typename Scalar>
struct MultibodyConstraintFwdDataTpl : ODEDataTpl<Scalar> {
  ALIGATOR_ODE_DATA_TYPEDEFS(Scalar, MultibodyConstraintFwdDynamicsTpl);

  using MultibodyConstraintCommon = MultibodyConstraintCommonTpl<Scalar>;
  using MultibodyConstraintCommonData =
      MultibodyConstraintCommonDataTpl<Scalar>;

  MultibodyConstraintFwdDataTpl(
      const MultibodyConstraintFwdDynamicsTpl<Scalar> &cont_dyn,
      const CommonModelDataContainer &container);

  const MultibodyConstraintCommonData *multibody_data_;
};

} // namespace dynamics
} // namespace aligator

#include "aligator/modelling/dynamics/multibody-constraint-fwd.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/modelling/dynamics/multibody-constraint-fwd.txx"
#endif
