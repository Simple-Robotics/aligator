/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/ode-abstract.hpp"
#include "aligator/modelling/dynamics/centroidal-fwd.hpp"

#include <proxsuite-nlp/modelling/spaces/cartesian-product.hpp>
#include <proxsuite-nlp/modelling/spaces/vector-space.hpp>
#include <proxsuite-nlp/modelling/spaces/multibody.hpp>

namespace aligator {
namespace dynamics {
template <typename Scalar> struct CentroidalKinematicsFwdDataTpl;

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
struct CentroidalKinematicsFwdDynamicsTpl : ODEAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ODEAbstractTpl<Scalar>;
  using BaseData = ODEDataTpl<Scalar>;
  using ContDataAbstract = ContinuousDynamicsDataTpl<Scalar>;
  using Data = CentroidalKinematicsFwdDataTpl<Scalar>;
  using Manifold = proxsuite::nlp::CartesianProductTpl<Scalar>;
  using ManifoldPtr = shared_ptr<Manifold>;
  using CentroidalPtr = shared_ptr<CentroidalFwdDynamicsTpl<Scalar>>;
  using Matrix3s = Eigen::Matrix<Scalar, 3, 3>;

  using Base::nu_;

  ManifoldPtr space_;
  const std::size_t nuc_;
  const std::size_t nv_;
  CentroidalPtr centroidal_;

  const Manifold &space() const { return *space_; }

  CentroidalKinematicsFwdDynamicsTpl(const ManifoldPtr &state, const size_t &nv,
                                     CentroidalPtr centroidal);

  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               BaseData &data) const;
  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) const;

  shared_ptr<ContDataAbstract> createData() const;
};

template <typename Scalar>
struct CentroidalKinematicsFwdDataTpl : ODEDataTpl<Scalar> {
  using Base = ODEDataTpl<Scalar>;

  shared_ptr<Base> centroidal_data_;

  CentroidalKinematicsFwdDataTpl(
      const CentroidalKinematicsFwdDynamicsTpl<Scalar> *model);
};

} // namespace dynamics
} // namespace aligator

#include "aligator/modelling/dynamics/centroidal-kinematics-fwd.hxx"
