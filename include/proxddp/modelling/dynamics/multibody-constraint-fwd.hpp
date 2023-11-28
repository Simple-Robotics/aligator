#pragma once

#include "proxddp/modelling/dynamics/ode-abstract.hpp"

#include <proxnlp/modelling/spaces/multibody.hpp>
#include <pinocchio/multibody/data.hpp>

#include <pinocchio/algorithm/proximal.hpp>

namespace proxddp {
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
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ODEAbstractTpl<Scalar>;
  using BaseData = ODEDataTpl<Scalar>;
  using ContDataAbstract = ContinuousDynamicsDataTpl<Scalar>;
  using Data = MultibodyConstraintFwdDataTpl<Scalar>;
  using RigidConstraintModelVector = PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(
      pinocchio::RigidConstraintModel);
  using RigidConstraintDataVector =
      PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(pinocchio::RigidConstraintData);
  using ProxSettings = pinocchio::ProximalSettingsTpl<Scalar>;
  using Manifold = proxnlp::MultibodyPhaseSpace<Scalar>;

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

  virtual void forward(const ConstVectorRef &x, const ConstVectorRef &u,
                       BaseData &data) const;
  virtual void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                        BaseData &data) const;

  shared_ptr<ContDataAbstract> createData() const;
};

template <typename Scalar>
struct MultibodyConstraintFwdDataTpl : ODEDataTpl<Scalar> {
  using Base = ODEDataTpl<Scalar>;
  using VectorXs = typename math_types<Scalar>::VectorXs;
  using MatrixXs = typename math_types<Scalar>::MatrixXs;
  using PinDataType = pinocchio::DataTpl<Scalar>;
  using RigidConstraintDataVector =
      PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(pinocchio::RigidConstraintData);

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
} // namespace proxddp

#include "proxddp/modelling/dynamics/multibody-constraint-fwd.hxx"

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxddp/modelling/dynamics/multibody-constraint-fwd.txx"
#endif
