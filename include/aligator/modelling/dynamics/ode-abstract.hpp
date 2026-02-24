#pragma once
/// @file ode-abstract.hpp
/// @brief Defines a class representing ODEs.
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2026 INRIA

#include "aligator/modelling/dynamics/continuous-dynamics-abstract.hpp"

namespace aligator {
namespace dynamics {
/// @brief   Base class for ODE dynamics \f$ \dot{x} = f(x, u) \f$.
/// @details  Formulated as a DAE (for ContinuousDynamicsAbstractTpl), this
/// class models \f[ f(x, u) - \dot{x}. \f]
template <typename _Scalar>
struct ODEAbstractTpl : ContinuousDynamicsAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  using Base = ContinuousDynamicsAbstractTpl<Scalar>;
  using Data = ContinuousDynamicsDataTpl<Scalar>;
  using Base::Base;
  using Base::nu_;
  using Base::space_;

  virtual ~ODEAbstractTpl() = default;

  /// Evaluate the ODE vector field: this returns the value of \f$\dot{x}\f$.
  virtual void forward(const ConstVectorRef &x, const ConstVectorRef &u,
                       Data &data) const = 0;

  /// Evaluate the vector field Jacobians.
  virtual void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                        Data &data) const = 0;

  /** Declare overrides **/

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &xdot, Data &data) const override;

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &xdot, Data &data) const override;
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct ODEAbstractTpl<context::Scalar>;
#endif

} // namespace dynamics
} // namespace aligator
