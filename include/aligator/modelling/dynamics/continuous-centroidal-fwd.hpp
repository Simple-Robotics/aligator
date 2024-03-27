/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/ode-abstract.hpp"
#include "aligator/modelling/contact-map.hpp"

#include <proxsuite-nlp/modelling/spaces/vector-space.hpp>

namespace aligator {
namespace dynamics {
template <typename Scalar> struct CentroidalFwdDataTpl;

/**
 * @brief   Nonlinear centroidal forward dynamics with smooth control.
 *
 * @details This is described in state-space \f$\mathcal{X} = T\mathcal{Q}\f$
 * (the phase space \f$x = (c,h,L,u)\f$ with c CoM position, h linear momentum,
 * L angular momentum and u forces) using the differential equation \f[ \dot{x}
 * = f(x, u) = F_x * x + F_u(x) * u \f] as described by the Newton-Euler law
 * of momentum. Control parameter is set to be the derivative of u to ensure
 * force smoothness.
 *
 */
template <typename _Scalar>
struct ContinuousCentroidalFwdDynamicsTpl : ODEAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ODEAbstractTpl<Scalar>;
  using BaseData = ODEDataTpl<Scalar>;
  using ContDataAbstract = ContinuousDynamicsDataTpl<Scalar>;
  using Data = ContinuousCentroidalFwdDataTpl<Scalar>;
  using Manifold = proxsuite::nlp::VectorSpaceTpl<Scalar>;
  using ManifoldPtr = shared_ptr<Manifold>;
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

  ContinuousCentroidalFwdDynamicsTpl(const ManifoldPtr &state,
                                     const double mass, const Vector3s &gravity,
                                     const ContactMap &contact_map,
                                     const int force_size);

  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               BaseData &data) const;
  void dForward(const ConstVectorRef &x, const ConstVectorRef &,
                BaseData &data) const;

  shared_ptr<ContDataAbstract> createData() const;
};

template <typename Scalar>
struct ContinuousCentroidalFwdDataTpl : ODEDataTpl<Scalar> {
  using Base = ODEDataTpl<Scalar>;
  using Matrix3s = Eigen::Matrix<Scalar, 3, 3>;

  Matrix3s Jtemp_;

  ContinuousCentroidalFwdDataTpl(
      const ContinuousCentroidalFwdDynamicsTpl<Scalar> *cont_dyn);
};

} // namespace dynamics
} // namespace aligator

#include "aligator/modelling/dynamics/continuous-centroidal-fwd.hxx"
