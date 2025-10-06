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
  using typename Base::Data;
  using DerivedData = ExplicitIntegratorDataTpl<Scalar>;

  using Base::dForward;
  using Base::forward;
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
      const xyz::polymorphic<ODEType> &cont_dynamics)
      : Base(cont_dynamics->space_, cont_dynamics->nu())
      , ode_(cont_dynamics) {}

  virtual ~ExplicitIntegratorAbstractTpl() = default;

  shared_ptr<Data> createData() const {
    return std::make_shared<DerivedData>(*this);
  }
};

template <typename _Scalar>
struct ExplicitIntegratorDataTpl : ExplicitDynamicsDataTpl<_Scalar> {
  using Scalar = _Scalar;
  using Model = ExplicitIntegratorAbstractTpl<Scalar>;
  using Base = ExplicitDynamicsDataTpl<Scalar>;
  using ODEData = ContinuousDynamicsDataTpl<Scalar>;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  shared_ptr<ODEData> continuous_data;
  VectorXs dx_;

  explicit ExplicitIntegratorDataTpl(const Model &integrator)
      : Base(integrator)
      , continuous_data(integrator.ode_->createData())
      , dx_(integrator.ndx2()) {
    dx_.setZero();
  }

  virtual ~ExplicitIntegratorDataTpl() = default;
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct ExplicitIntegratorAbstractTpl<context::Scalar>;
extern template struct ExplicitIntegratorDataTpl<context::Scalar>;
#endif
} // namespace dynamics
} // namespace aligator
