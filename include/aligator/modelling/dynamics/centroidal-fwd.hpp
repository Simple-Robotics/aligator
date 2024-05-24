/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/ode-abstract.hpp"
#include "aligator/modelling/contact-map.hpp"

#include <proxsuite-nlp/modelling/spaces/vector-space.hpp>

namespace aligator {
namespace dynamics {
template <typename Scalar> struct CentroidalFwdDataTpl;

/**
 * @brief   Nonlinear centroidal forward dynamics.
 *
 * @details This is described in state-space \f$\mathcal{X} = T\mathcal{Q}\f$
 * (the phase space \f$x = (c,h,L)\f$ with c CoM position, h linear momentum)
 * and L angular momentum) using the differential equation \f[ \dot{x}
 * = f(x, u) = F_x * x + F_u(x) * u \f] as described by the Newton-Euler law
 * of momentum.
 *
 */
template <typename _Scalar>
struct CentroidalFwdDynamicsTpl : ODEAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ODEAbstractTpl<Scalar>;
  using BaseData = ODEDataTpl<Scalar>;
  using ContDataAbstract = ContinuousDynamicsDataTpl<Scalar>;
  using Data = CentroidalFwdDataTpl<Scalar>;
  using Manifold = proxsuite::nlp::VectorSpaceTpl<Scalar>;
  using ManifoldPtr = xyz::polymorphic<Manifold>;
  using Matrix3s = Eigen::Matrix<Scalar, 3, 3>;
  using ContactMap = ContactMapTpl<Scalar>;

  using Base::nu_;

  ManifoldPtr space_;
  std::size_t nk_;
  double mass_;
  Vector3s gravity_;
  ContactMap contact_map_;
  int force_size_;

  const Manifold &space() const { return *space_; }

  CentroidalFwdDynamicsTpl(const ManifoldPtr &state, const double mass,
                           const Vector3s &gravity,
                           const ContactMap &contact_map, const int force_size);

  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               BaseData &data) const;
  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) const;

  shared_ptr<ContDataAbstract> createData() const;
};

template <typename Scalar> struct CentroidalFwdDataTpl : ODEDataTpl<Scalar> {
  using Base = ODEDataTpl<Scalar>;
  using Matrix3s = Eigen::Matrix<Scalar, 3, 3>;

  Matrix3s Jtemp_;

  CentroidalFwdDataTpl(const CentroidalFwdDynamicsTpl<Scalar> *cont_dyn);
};

} // namespace dynamics
} // namespace aligator

#include "aligator/modelling/dynamics/centroidal-fwd.hxx"
