/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/ode-abstract.hpp"
#include "aligator/modelling/contact-map.hpp"

#include <proxsuite-nlp/modelling/spaces/cartesian-product.hpp>
#include <proxsuite-nlp/modelling/spaces/vector-space.hpp>
#include <proxsuite-nlp/modelling/spaces/multibody.hpp>
#include <pinocchio/multibody/model.hpp>

namespace aligator {
namespace dynamics {
template <typename Scalar> struct KinodynamicsFwdDataTpl;

/**
 * @brief   Nonlinear centroidal and kinematics forward dynamics.
 *
 * @details Cartesian product of centroidal dynamics and Pinocchio kinematics.
 * This is described in state-space \f$\mathcal{X} = T\mathcal{Q}\f$
 * (the phase space \f$x = (c,h,L,q,v)\f$ with c CoM position, h linear
 * momentum, L angular momentum, q kinematics pose and v joint velocity) using
 * the differential equation \f[ \dot{c,h,l} = f(c,h,l,\lambda) = F_x * x
 * + F_{lambda}}(x) * lambda \f]
 * with f contact forces as described by the Newton-Euler law of momentum.
 * Kinematics component is subjected to \f$\begin{bmatrix} \dot{q} \\ \dot{v}
 * \end{bmatrix} = \begin{bmatrix} v \\ a \end{bmatrix}\f$ with a commanded
 * joint acceleration.
 *
 */
template <typename _Scalar>
struct KinodynamicsFwdDynamicsTpl : ODEAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ODEAbstractTpl<Scalar>;
  using BaseData = ODEDataTpl<Scalar>;
  using ContDataAbstract = ContinuousDynamicsDataTpl<Scalar>;
  using Data = KinodynamicsFwdDataTpl<Scalar>;
  using Manifold = proxsuite::nlp::CartesianProductTpl<Scalar>;
  using ManifoldPtr = shared_ptr<Manifold>;
  using Model = pinocchio::ModelTpl<Scalar>;
  using Matrix3s = Eigen::Matrix<Scalar, 3, 3>;
  using ContactMap = ContactMapTpl<Scalar>;

  using Base::nu_;

  ManifoldPtr space_;
  Model pin_model_;
  double mass_;
  Vector3s gravity_;
  ContactMap contact_map_;

  const Manifold &space() const { return *space_; }

  KinodynamicsFwdDynamicsTpl(const ManifoldPtr &state, const Model &model,
                             const Vector3s &gravity,
                             const ContactMap &contact_map);

  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               BaseData &data) const;
  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) const;

  shared_ptr<ContDataAbstract> createData() const;
};

template <typename Scalar> struct KinodynamicsFwdDataTpl : ODEDataTpl<Scalar> {
  using Base = ODEDataTpl<Scalar>;
  using PinData = pinocchio::DataTpl<Scalar>;
  using Force = pinocchio::ForceTpl<Scalar>;
  using Matrix6Xs = typename math_types<Scalar>::Matrix6Xs;
  using Matrix3s = Eigen::Matrix<Scalar, 3, 3>;

  PinData pin_data_;
  Matrix3s Jtemp_;
  Force hdot_;
  Matrix6Xs dh_dq_;
  Matrix6Xs dhdot_dq_;
  Matrix6Xs dhdot_dv_;
  Matrix6Xs dhdot_da_;

  KinodynamicsFwdDataTpl(const KinodynamicsFwdDynamicsTpl<Scalar> *model);
};

} // namespace dynamics
} // namespace aligator

#include "aligator/modelling/dynamics/kinodynamics-fwd.hxx"
