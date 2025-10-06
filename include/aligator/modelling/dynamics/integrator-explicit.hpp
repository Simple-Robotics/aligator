#pragma once
/// @file integrator-explicit.hpp
/// @brief  Base definitions for explicit integrators.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA

#include "aligator/core/explicit-dynamics.hpp"
#include "aligator/modelling/dynamics/ode-abstract.hpp"

namespace aligator {
namespace dynamics {

/**
 * @brief   Explicit integrators \f$x_{k+1} = f(x_k, u_k) \f$.
 * @details This class of integrator mostly applies to integrating ODE models
 * \f$\dot{x} = \phi(x,u)\f$. This class is separate from IntegratorAbstractTpl
 * and not a child class; this ensures there is no diamond inheritance problem.
 */
template <typename _Scalar>
struct ExplicitIntegratorAbstractTpl : ExplicitDynamicsModelTpl<_Scalar> {
  using Scalar = _Scalar;
  using ODEType = ODEAbstractTpl<Scalar>;
  using Base = ExplicitDynamicsModelTpl<Scalar>;
  using Data = ExplicitIntegratorDataTpl<Scalar>;

  using Base::computeJacobians;
  using Base::evaluate;
  using Base::ndx1;
  using Base::ndx2;
  using Base::nu;
  using Base::space_next_;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  xyz::polymorphic<ODEType> ode_;

  template <typename U> U *getDynamics() { return dynamic_cast<U *>(&*ode_); }
  template <typename U> const U *getDynamics() const {
    return dynamic_cast<const U *>(&*ode_);
  }

  explicit ExplicitIntegratorAbstractTpl(
      const xyz::polymorphic<ODEType> &cont_dynamics);
  virtual ~ExplicitIntegratorAbstractTpl() = default;

  shared_ptr<DynamicsDataTpl<Scalar>> createData() const;
};

template <typename _Scalar>
struct ExplicitIntegratorDataTpl : ExplicitDynamicsDataTpl<_Scalar> {
  using Scalar = _Scalar;
  using Base = ExplicitDynamicsDataTpl<Scalar>;
  using ODEData = ContinuousDynamicsDataTpl<Scalar>;
  shared_ptr<ODEData> continuous_data;

  explicit ExplicitIntegratorDataTpl(
      const ExplicitIntegratorAbstractTpl<Scalar> *integrator);
  virtual ~ExplicitIntegratorDataTpl() = default;

  using Base::dx_;
  using Base::xnext_;
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct ExplicitIntegratorAbstractTpl<context::Scalar>;
extern template struct ExplicitIntegratorDataTpl<context::Scalar>;
#endif
} // namespace dynamics
} // namespace aligator
