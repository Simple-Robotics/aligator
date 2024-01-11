/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/ode-abstract.hpp"

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
  using ManifoldPtr = shared_ptr<Manifold>;
  using Matrix3s = Eigen::Matrix<Scalar, 3, 3>;

  using Base::nu_;

  ManifoldPtr space_;
  const std::size_t nk_;
  const double mass_;
  Vector3s gravity_;
  std::vector<bool> active_contacts_;
  StdVectorEigenAligned<Vector3s> contact_points_;

  const Manifold &space() const { return *space_; }

  CentroidalFwdDynamicsTpl(const ManifoldPtr &state, const int &nk,
                           const double &mass);

  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               BaseData &data) const;
  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) const;

  shared_ptr<ContDataAbstract> createData() const;
};

template <typename Scalar> struct CentroidalFwdDataTpl : ODEDataTpl<Scalar> {
  using Base = ODEDataTpl<Scalar>;

  CentroidalFwdDataTpl(const CentroidalFwdDynamicsTpl<Scalar> *cont_dyn);
};

} // namespace dynamics
} // namespace aligator

#include "aligator/modelling/dynamics/centroidal-fwd.hxx"
