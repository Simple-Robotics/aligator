#pragma once
/// @file ode-abstract.hpp
/// @brief Defines a class representing ODEs.
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA

#include "aligator/modelling/dynamics/continuous-base.hpp"

namespace aligator {
namespace dynamics {
/** @brief   Base class for ODE dynamics \f$ \dot{x} = f(x, u) \f$.
 * @details  Formulated as a DAE (for ContinuousDynamicsAbstractTpl), this class
 * models \f[ f(x, u) - \dot{x}. \f]
 */
template <typename _Scalar>
struct ODEAbstractTpl : ContinuousDynamicsAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  using Base = ContinuousDynamicsAbstractTpl<Scalar>;
  using ContDataAbstract = ContinuousDynamicsDataTpl<Scalar>;
  using Base::Base;
  using Base::nu_;
  using Base::space_;
  using ODEData = ODEDataTpl<Scalar>;

  virtual ~ODEAbstractTpl() = default;

  /// Evaluate the ODE vector field: this returns the value of \f$\dot{x}\f$.
  virtual void forward(const ConstVectorRef &x, const ConstVectorRef &u,
                       ODEData &data) const = 0;

  /// Evaluate the vector field Jacobians.
  virtual void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                        ODEData &data) const = 0;

  /** Declare overrides **/

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &xdot,
                ContDataAbstract &data) const override;

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &xdot,
                        ContDataAbstract &data) const override;

  virtual shared_ptr<ContDataAbstract> createData() const override;
};

template <typename _Scalar>
struct ODEDataTpl : ContinuousDynamicsDataTpl<_Scalar> {
  using Scalar = _Scalar;
  using VectorXs = typename math_types<Scalar>::VectorXs;
  using Base = ContinuousDynamicsDataTpl<Scalar>;

  /// Time derivative \f$\dot{x} = f(x, u)\f$, output of ODE model
  VectorXs xdot_;

  ODEDataTpl(const int ndx, const int nu);
  virtual ~ODEDataTpl() = default;
};

} // namespace dynamics
} // namespace aligator

#include "aligator/modelling/dynamics/ode-abstract.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/modelling/dynamics/ode-abstract.txx"
#endif
